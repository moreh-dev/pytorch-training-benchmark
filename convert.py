import torch
import torch.nn as nn
import transformer_engine.pytorch as te

from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention

from llama import LLaMAConfig, Attention, MLP, RMSNorm, Fp8LLaMA, Fp8LLaMABlock


# LlamaSdpaAttention -> llama.Attention
def convert_attention(hf_attn: nn.Module, embedding_dim, num_heads,
                      num_kv_heads) -> nn.Module:
    new_attn = Attention(embedding_dim=embedding_dim,
                         num_heads=num_heads,
                         num_kv_heads=num_kv_heads)
    new_attn.in_proj.weight.data.copy_(hf_attn.qkv_proj.weight.data)
    new_attn.out_proj.weight.data.copy_(hf_attn.o_proj.weight.data)
    # new_attn.rotary_emb = LlamaRotaryEmbedding()

    return new_attn


# LlamaMLP -> llama.MLP
def convert_mlp(hf_mlp: nn.Module, embedding_dim, hidden_dim) -> nn.Module:

    new_mlp = MLP(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    new_mlp.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data)
    new_mlp.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data)
    new_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)
    # new_mlp.act_fn = SiLU()

    return new_mlp


# LlamaRMSNorm -> llama.RMSNorm
def convert_rmsnorm(hf_rmsnorm: nn.Module) -> nn.Module:
    hidden_size = hf_rmsnorm.weight.shape[0]
    eps = hf_rmsnorm.variance_epsilon
    new_rmsnorm = RMSNorm(embedding_dim=hidden_size, eps=eps)
    new_rmsnorm.weight.data.copy_(hf_rmsnorm.weight.data)

    return new_rmsnorm


def check_multihead_attention(layer: nn.Module):
    if layer is None or not isinstance(layer, te.MultiheadAttention):
        raise Exception("self_attention is None or not MultiheadAttention")

    if layer.layernorm_qkv is None or not isinstance(layer.layernorm_qkv,
                                                     te.LayerNormLinear):
        raise Exception(
            "self_attention.layernorm_qkv is None or not LayerNormLinear")

    if layer.core_attention is None or not isinstance(layer.core_attention,
                                                      te.DotProductAttention):
        raise Exception(
            "self_attention.core_attention is None or not DotProductAttention")

    if layer.core_attention.fused_attention is None or not isinstance(
            layer.core_attention.fused_attention, te.FusedAttention):
        raise Exception(
            "self_attention.core_attention.fused_attention is None or not FusedAttention"
        )

    if layer.proj is None or not isinstance(layer.proj, te.Linear):
        raise Exception("self_attention.proj is None or not Linear")


# LlamaDecoderLayer -> llama.Fp8LLaMABlock
def convert_block(hf_layer: nn.Module, embedding_dim, num_heads, num_kv_heads,
                  hidden_dim, eps) -> nn.Module:
    new_block = Fp8LLaMABlock(embedding_dim=embedding_dim,
                              hidden_dim=hidden_dim,
                              num_heads=num_heads,
                              num_kv_heads=num_kv_heads,
                              eps=eps)

    # Maybe need to update activation to SiLU?

    # Multihead Attention
    check_multihead_attention(new_block.self_attention)

    input_layernorm_w = hf_layer.input_layernorm.weight  # shape: [8192]

    q_w = hf_layer.self_attn.q_proj.weight  # shape: [8192, 8192]
    k_w = hf_layer.self_attn.k_proj.weight  # shape: [8192, 1024]
    v_w = hf_layer.self_attn.v_proj.weight  # shape: [8192, 1024]
    o_w = hf_layer.self_attn.o_proj.weight  # shape: [8192, 8192]
    fused_qkv_w = torch.cat([q_w, k_w, v_w], dim=-1)

    new_block.self_attention.layernorm_qkv.layer_norm_weight.data.copy_(
        input_layernorm_w)
    new_block.self_attention.core_attention.in_proj.weight.data.copy_(
        fused_qkv_w)
    new_block.self_attention.core_attention.out_proj.weight.data.copy_(o_w)

    # LayerNormMLP
    up_w = hf_layer.mlp.up_proj.weight  # shape: [8192, 28672]
    gate_w = hf_layer.mlp.gate_proj.weight  # shape: [8192, 28672]
    down_w = hf_layer.mlp.down_proj.weight  # shape: [8192, 8192]

    new_block.attn_norm = convert_rmsnorm(hf_layer.layer_norm_weight)

    new_block.layernorm_mlp.up_proj.weight.data.copy_(up_w)
    new_block.layernorm_mlp.gate_proj.weight.data.copy_(gate_w)
    new_block.layernorm_mlp.down_proj.weight.data.copy_(down_w)

    new_block.mlp = convert_mlp(hf_layer.mlp, embedding_dim, hidden_dim)
    new_block.mlp_norm = convert_rmsnorm(hf_layer.post_attention_layernorm)

    return new_block


def convert_model(hf_model: nn.Module, config: dict) -> nn.Module:
    new_model = Fp8LLaMA(vocab_size=config['vocab_size'],
                         embedding_dim=config['embedding_dim'],
                         hidden_dim=config['hidden_dim'],
                         num_layers=config['num_layers'],
                         num_heads=config['num_heads'],
                         num_kv_heads=config['num_kv_heads'],
                         max_seq_len=config['max_seq_len'],
                         eps=config['eps'])

    new_model.embedding.weight.data.copy_(
        hf_model.model.embed_tokens.weight.data)

    for i in range(config['num_layers']):
        hf_layer = hf_model.model.layers[i]
        new_layer = convert_block(hf_layer,
                                  embedding_dim=config['embedding_dim'],
                                  num_heads=config['num_heads'],
                                  num_kv_heads=config['num_kv_heads'],
                                  hidden_dim=config['hidden_dim'],
                                  eps=config['eps'])
        new_model.layers[i] = new_layer

    new_model.norm = convert_rmsnorm(hf_model.model.norm)
    new_model.rotary_emb = hf_model.model.rotary_emb
    new_model.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)

    return new_model


if __name__ == "__main__":
    hf_model = LlamaForCausalLM.from_pretrained(
        "/vast/huggingface/hub/models--regisss--llama2-70b-fused-qkv-mlperf/snapshots/647cb0c8858ddefd10231a20ddfa68e4eb5e850e/",
        use_cache=False,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=False,
        max_position_embeddings=8192,
        local_files_only=True)
    config = {
        'vocab_size': 8192,
        'embedding_dim': 8192,
        'hidden_dim': 8192,
        'num_layers': 80,
        'num_heads': 64,
        'num_kv_heads': 8,
        'max_seq_len': 8192,
        'eps': 1e-5,
    }

    model = convert_model(hf_model, config)

    for name, module in model.named_modules():
        print(f'name={name}, module={module}')
        #if isinstance(module, LlamaAttention):
        #    replace_modules(module)
