import fire
import json
import torch
from dataclasses import asdict
from llama import LLaMAConfig, LLaMA, Fp8LLaMA
from mistral import MistralConfig, Mistral, Fp8Mistral
from transformers import AutoModelForCausalLM


def copy_weight(tensor_from, tensor_to):
    if tensor_to.requires_grad:
        with torch.no_grad():
            tensor_to.data.copy_(tensor_from.detach().cpu())
        tensor_to.requires_grad = True
    else:
        tensor_to.data.copy_(tensor_from.detach().cpu())


def convert_model(model: torch.nn.Module, hf_model_path: str,
                  config: dict) -> torch.nn.Module:
    torch.cuda.empty_cache()

    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        use_cache=False,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=False,
        max_position_embeddings=config['max_seq_len'],
        local_files_only=True,
        trust_remote_code=True).to("cpu")

    print(f'Loaded model from {hf_model_path}')
    print(f'{len(hf_model.model.layers)} layers detected')

    # CPU RAM 활용
    hf_embed_tokens_w = hf_model.model.embed_tokens.weight.to(
        torch.float32).cpu()
    hf_lm_head_w = hf_model.lm_head.weight.to(torch.float32).cpu()
    hf_norm_w = hf_model.model.norm.weight.to(torch.float32).cpu()

    copy_weight(hf_embed_tokens_w, model.embedding.weight)
    copy_weight(hf_lm_head_w, model.norm_lm_head.weight)
    copy_weight(hf_norm_w, model.norm_lm_head.layer_norm_weight)

    del hf_embed_tokens_w, hf_lm_head_w, hf_norm_w
    torch.cuda.empty_cache()

    for i in range(config['num_layers']):
        hf_layer = hf_model.model.layers[i]
        new_layer = model.layers[i]

        # Multihead Attention
        hf_self_attn = hf_layer.self_attn
        new_self_attention = new_layer.self_attention

        hf_self_attn_input_layernorm_w = hf_layer.input_layernorm.weight.to(
            torch.float32).cpu()
        hf_self_attn_qkv_w = hf_self_attn.qkv_proj.weight.to(
            torch.float32).cpu()
        hf_self_attn_o_w = hf_self_attn.o_proj.weight.to(torch.float32).cpu()

        copy_weight(hf_self_attn_input_layernorm_w,
                    new_self_attention.layernorm_qkv.layer_norm_weight)
        copy_weight(hf_self_attn_qkv_w,
                    new_self_attention.layernorm_qkv.weight)
        copy_weight(hf_self_attn_o_w, new_self_attention.proj.weight)

        del hf_self_attn_input_layernorm_w, hf_self_attn_qkv_w, hf_self_attn_o_w
        torch.cuda.empty_cache()

        # LayerNormMLP
        hf_mlp = hf_layer.mlp
        new_mlp = new_layer.layernorm_mlp

        hf_mlp_post_attn_layernorm_w = hf_layer.post_attention_layernorm.weight.to(
            torch.float32).cpu()
        hf_mlp_fc1_w = torch.cat(
            [hf_mlp.gate_proj.weight, hf_mlp.up_proj.weight],
            dim=0).to(torch.float32).cpu()
        hf_mlp_fc2_w = hf_mlp.down_proj.weight.to(torch.float32).cpu()

        copy_weight(hf_mlp_post_attn_layernorm_w, new_mlp.layer_norm_weight)
        copy_weight(hf_mlp_fc1_w, new_mlp.fc1_weight)
        copy_weight(hf_mlp_fc2_w, new_mlp.fc2_weight)

        del hf_mlp_post_attn_layernorm_w, hf_mlp_fc1_w, hf_mlp_fc2_w
        torch.cuda.empty_cache()

        print(f'Layer #{i} conversion completed')

    del hf_model
    torch.cuda.empty_cache()
    print(f'Whole model conversion completed')

    return model


def convert_and_save_model(config_file: str,
                           model_name: str,
                           hf_model_path: str,
                           save_path: str,
                           enable_fp8: bool = True):
    with open(config_file) as f:
        config = json.load(f)

    if model_name == "llama":
        model_config = LLaMAConfig(**config)
    elif model_name == "mistral":
        model_config = MistralConfig(**config)
    else:
        print(
            "Model not supported. Please pass either 'llama' or 'mistral' as a parameter."
        )
        return

    if enable_fp8:
        if model_name == "llama":
            model = Fp8LLaMA(**asdict(model_config))
        elif model_name == "mistral":
            model = Fp8Mistral(**asdict(model_config))
    else:
        if model_name == "llama":
            model = LLaMA(**asdict(model_config))
        elif model_name == "mistral":
            model = Mistral(**asdict(model_config))

    model.to("cpu")  # CPU에서 변환 진행

    model = convert_model(model, hf_model_path, config)

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')


if __name__ == '__main__':
    fire.Fire(convert_and_save_model)
