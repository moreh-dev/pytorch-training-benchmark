import os
import fire
import json
import time
import numpy as np
from dataclasses import asdict
from datasets import load_dataset
from contextlib import nullcontext
from functools import partial
from llama import LLaMAConfig, LLaMA, LLaMABlock, Fp8LLaMA, Fp8LLaMABlock
from mistral import MistralConfig, Mistral, MistralBlock, Fp8Mistral, Fp8MistralBlock

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import destroy_process_group
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)

from torch.utils.data import DataLoader, IterableDataset
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from transformer_engine.pytorch.distributed import prepare_te_modules_for_fsdp

from transformers import AutoModelForCausalLM


def copy_weight(tensor_from, tensor_to):
    if tensor_to.requires_grad:
        with torch.no_grad():
            tensor_to.data.copy_(tensor_from.detach())
        tensor_to.requires_grad = True
    else:
        tensor_to.data.copy_(tensor_from.detach())


def convert_model(model: torch.nn.Module, hf_model_path: str,
                  config: dict) -> torch.nn.Module:
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        use_cache=False,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=False,
        max_position_embeddings=config['max_seq_len'],
        local_files_only=True,
        trust_remote_code=True)

    print(f'Loaded model from {hf_model_path}')
    print(f'{len(hf_model.model.layers)} layers detected')

    # missing position_encoding: torch.Size([8192, 1, 1, 128]), torch.bfloat16

    hf_embed_tokens_w = hf_model.model.embed_tokens.weight.to(torch.float32)
    hf_lm_head_w = hf_model.lm_head.weight.to(torch.float32)
    hf_norm_w = hf_model.model.norm.weight.to(torch.float32)

    copy_weight(hf_embed_tokens_w, model.embedding.weight)
    copy_weight(hf_lm_head_w, model.norm_lm_head.weight)
    copy_weight(hf_norm_w, model.norm_lm_head.layer_norm_weight)

    for i in range(config['num_layers']):
        hf_layer = hf_model.model.layers[i]
        new_layer = model.layers[i]

        # Multihead Attention
        hf_self_attn = hf_layer.self_attn
        new_self_attention = new_layer.self_attention

        hf_self_attn_input_layernorm_w = hf_layer.input_layernorm.weight.to(
            torch.float32)
        hf_self_attn_qkv_w = hf_self_attn.qkv_proj.weight.to(torch.float32)
        hf_self_attn_o_w = hf_self_attn.o_proj.weight.to(torch.float32)

        copy_weight(hf_self_attn_input_layernorm_w,
                    new_self_attention.layernorm_qkv.layer_norm_weight)
        copy_weight(hf_self_attn_qkv_w,
                    new_self_attention.layernorm_qkv.weight)
        copy_weight(hf_self_attn_o_w, new_self_attention.proj.weight)

        # LayerNormMLP
        hf_mlp = hf_layer.mlp
        new_mlp = new_layer.layernorm_mlp

        hf_mlp_post_attn_layernorm_w = hf_layer.post_attention_layernorm.weight.to(
            torch.float32)
        hf_mlp_fc1_w = torch.cat(
            [hf_mlp.gate_proj.weight, hf_mlp.up_proj.weight],
            dim=0).to(torch.float32)
        hf_mlp_fc2_w = hf_mlp.down_proj.weight.to(torch.float32)

        copy_weight(hf_mlp_post_attn_layernorm_w, new_mlp.layer_norm_weight)
        copy_weight(hf_mlp_fc1_w, new_mlp.fc1_weight)
        copy_weight(hf_mlp_fc2_w, new_mlp.fc2_weight)

        print(f'Layer #{i} conversion completed')
    print(f'Whole model conversion completed')

    return model


class HFIterableDataset(IterableDataset):

    def __init__(self, hf_dataset):
        super().__init__()
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):

        for item in self.dataset:
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
            labels = torch.tensor(item["labels"], dtype=torch.long)
            yield {"input_ids": input_ids, "labels": labels}


def create_train_data_loader(train_dataset_path, world_size, batch_size,
                             num_iteration, model_config):
    hf_dataset = load_dataset("parquet",
                              data_files=train_dataset_path,
                              split="train")
    hf_iterable_dataset = HFIterableDataset(hf_dataset)

    data_loader = DataLoader(hf_iterable_dataset,
                             batch_size=batch_size,
                             num_workers=world_size,
                             pin_memory=True,
                             shuffle=False)
    return data_loader


def train(
        config_file: str = "conigs/llama-3.1-70b.moreh.json",
        model_name: str = "llama",
        model_path: str = "/path/to/model",
        dataset_path: str = "/path/to/dataset",
        batch_size: int = 1,
        num_iteration: int = 128,
        grad_accumlate_pre_steps: int = 8,  # steps to accumlate gradient
        reduce_pre_steps: int = 32,  # steps to do all reduce
        enable_fp8: bool = True,
        enable_compile: bool = False,
        seed: int = 1024  # to ensure reproducible
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    assert rank == local_rank, "This script is intended to run on single node for testing"
    world_size = int(os.environ["WORLD_SIZE"])
    if enable_fp8:
        enable_compile = False
        if rank == 0:
            print(
                'PyTorch compile currently doesn\'t work with Transformer Engine.'
            )
    # Construct process group
    if local_rank == 0:
        print("Initing communication")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    assert dist.get_rank() == local_rank
    # Configure training setup
    if local_rank == 0:
        print("Using config", config_file)
    with open(config_file) as f:
        config = json.load(f)

    if model_name == "llama":
        model_config = LLaMAConfig(**config)
    elif model_name == "mistral":
        model_config = MistralConfig(**config)
    else:
        print(
            "model not supported. please pass either llama or mistral as param"
        )
    if local_rank == 0:
        print("Creating model with config: ", model_config)

    if enable_fp8:  # add more model
        if model_name == "llama":
            layer_class = Fp8LLaMABlock
            model = Fp8LLaMA(**asdict(model_config))
        elif model_name == "mistral":
            layer_class = Fp8MistralBlock
            model = Fp8Mistral(**asdict(model_config))
    else:
        if model_name == "llama":
            layer_class = LLaMABlock
            model = LLaMA(**asdict(model_config))
        elif model_name == "mistral":
            layer_class = MistralBlock
            model = Mistral(**asdict(model_config))

    model = convert_model(model, model_path, config)

    model_config.estimate_flops_per_token(
        model, batch_size)  # Need to calculate before wrapping in FSDP

    if local_rank == 0:
        print(
            f"Loaded model on CPU with number of parameters: {sum(p.numel() for p in model.parameters())/1e9:.2f}B"
        )
        print(f"Original Model:\n{model}")

    # FSDP
    model = FSDP(model,
                 device_id=local_rank,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                                reduce_dtype=torch.bfloat16,
                                                buffer_dtype=torch.bfloat16),
                 auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                          transformer_layer_cls={layer_class}),
                 use_orig_params=True)
    if enable_fp8:
        prepare_te_modules_for_fsdp(model)
        fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
        fp8_recipe = DelayedScaling(fp8_format=fp8_format,
                                    amax_history_len=16,
                                    amax_compute_algo='max')
        all_worker = dist.new_group(backend='nccl')

    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1.0)

    # Print out allocated device memory
    pre_mem_use = torch.cuda.memory_allocated(
        device=f"cuda:{local_rank}") * 1e-6
    flops_per_iter = model_config.flops_per_token * (batch_size *
                                                     model_config.max_seq_len)
    if local_rank == 0:
        print(f"GPU memory use = {pre_mem_use}MB")
        print("TFLOP per iteration:", flops_per_iter / 1e12)
    # PyTorch compile
    if enable_compile:
        if rank == 0:
            print(f'Compiling model....')
        model = torch.compile(model)

    ddp_loss = torch.zeros(2, device=local_rank)
    model.train()
    iter_times = []
    warm_up = num_iteration / 2

    data_loader = create_train_data_loader(dataset_path, world_size,
                                           batch_size, num_iteration,
                                           model_config)
    last_time = time.time()

    for step_idx, data_batch in enumerate(data_loader):
        input = data_batch["input_ids"]
        labels = data_batch["labels"]

        input = input.to(local_rank)
        labels = labels.to(local_rank)
        fp8_context = nullcontext() if not enable_fp8 else te.fp8_autocast(
            enabled=enable_fp8, fp8_recipe=fp8_recipe, fp8_group=all_worker)
        with torch.amp.autocast('cuda', torch.bfloat16), fp8_context:
            weight_cache = enable_fp8 and (step_idx % grad_accumlate_pre_steps
                                           == 0)
            logits = model(input, is_first_microbatch=weight_cache)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            loss /= grad_accumlate_pre_steps

        loss.backward()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += input.size(0)

        if (step_idx + 1) % grad_accumlate_pre_steps == 0:
            # https://github.com/foundation-model-stack/fms-fsdp/blob/0fdb43dcfd31ab093f8d873b58b0b531dd0818b1/fms_fsdp/utils/train_utils.py#L94
            # https://github.com/foundation-model-stack/foundation-model-stack/blob/d55a9f2ade65ef4157cdfd928300874e2348e5d0/fms/training/trainer.py#L36
            model.clip_grad_norm_(1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if (step_idx + 1) % reduce_pre_steps == 0:
            dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

        if rank == 0:
            current_time = time.time()
            iter_time = current_time - last_time
            token_per_sec = (batch_size * model_config.max_seq_len) / iter_time
            if step_idx > warm_up:
                print(
                    f"Step: {step_idx}; TFLOP/s: {flops_per_iter/iter_time/1e12}; iteration time: {iter_time}; token per second: {token_per_sec}"
                )
                iter_times.append(iter_time)
            else:
                print(
                    f"warming up iter: {step_idx}/{warm_up}; TFLOP/s: {flops_per_iter/iter_time/1e12}; iteration time: {iter_time}; token per second: {token_per_sec}"
                )
            last_time = current_time

        if (step_idx + 1) == num_iteration:
            break

    if rank == 0:
        iter_times = np.array(iter_times)
        avg_iter_time = np.mean(iter_times)
        print("Avg token per second:",
              (batch_size * model_config.max_seq_len) / avg_iter_time)
        print("Avg iter time:", avg_iter_time)
        print("TFLOP per iteration:", flops_per_iter / 1e12)
        print("Avg TFLOP/s,", flops_per_iter / avg_iter_time / 1e12)
        peak_memory = torch.cuda.max_memory_allocated(
            device=f"cuda:{local_rank}") * 1e-6
        print(f"Peak memory use = {peak_memory}MB")

    torch.cuda.empty_cache()
    dist.barrier()
    destroy_process_group()


if __name__ == '__main__':
    fire.Fire(train)
