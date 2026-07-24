#!/usr/bin/env bash
# 估算当前配置下一次 GRPO run 两张卡各需要多少显存（纯配置推算，不依赖本机状态）。
set -euo pipefail

config_path="${GRPO_CONFIG:-/home/2025user/zyp/work/2607_trl_swe_agent/configs/grpo_swegym_qwen2_5_coder_7b_lora.yaml}"
.venv/bin/python - "$config_path" <<'EOF'
import json
import sys
from pathlib import Path

from swe_agent.config import load_config

GB = 1024**3
config, _, _ = load_config(sys.argv[1])
model_cfg = json.loads((Path(config.model.model_path) / "config.json").read_text())

H = model_cfg["hidden_size"]
L = model_cfg["num_hidden_layers"]
I = model_cfg["intermediate_size"]
V = model_cfg["vocab_size"]
heads = model_cfg["num_attention_heads"]
kv_heads = model_cfg["num_key_value_heads"]
kv_dim = H * kv_heads // heads
tied = model_cfg.get("tie_word_embeddings", False)
dtype_bytes = 2 if config.model.dtype == "bfloat16" else 4

# --- 模型参数 ---
embed = V * H
per_layer = (
    H * H + H                      # q proj + bias
    + 2 * (H * kv_dim + kv_dim)    # k, v proj + bias
    + H * H                        # o proj
    + 3 * H * I                    # mlp gate/up/down
    + 2 * H                        # rms norms
)
params = embed * (1 if tied else 2) + per_layer * L + H  # final norm

# --- LoRA ---
r, alpha_modules = config.peft.rank, config.peft.target_modules
lora = 0
for m in alpha_modules:
    if m in ("q_proj", "o_proj"):
        lora += r * (H + H)
    elif m in ("k_proj", "v_proj"):
        lora += r * (H + kv_dim)
lora *= L

# --- 训练侧显存（trainer 卡） ---
weights_gb = params * dtype_bytes / GB
lora_gb = lora * dtype_bytes / GB
adam_gb = lora * 8 / GB                      # AdamW 两个 fp32 动量
grad_gb = lora_gb
T_prompt = config.chat.max_prompt_length
T_comp = config.generation.max_completion_length
act_gb = L * (T_prompt + T_comp) * H * 2 / GB + 4  # grad ckpt 边界激活 + 重算峰值
logits_bf16 = T_comp * V * 2 / GB            # completion 段 logits（logits_to_keep）
logits_fp32 = T_comp * V * 4 / GB            # accelerate convert_to_fp32 峰值
logits_extra = logits_fp32                   # log_softmax 的第二个 fp32 副本（最坏情况）

steady = weights_gb + lora_gb + adam_gb + grad_gb + act_gb
peak_typ = steady + logits_bf16 + logits_fp32
peak_worst = peak_typ + logits_extra

# --- vLLM server 卡 ---
gpu_total = 80.0  # A100-SXM4-80GB
server_gb = config.vllm.gpu_memory_utilization * gpu_total

p = lambda x: f"{x:6.1f}"
print("=" * 56)
print("GRPO 显存估算（按配置推算，非实测）")
print("=" * 56)
print(f"模型: {config.model.provenance_id}  参数量: {params/1e9:.2f}B ({config.model.dtype})")
print(f"序列预算: prompt {T_prompt} + completion {T_comp}")
print()
print(f"[vLLM server 卡]  gpu_memory_utilization={config.vllm.gpu_memory_utilization}")
print(f"  固定占用 ≈ {p(server_gb)} GB（权重 + KV cache 池，启动即占满该比例）")
print()
print("[trainer 卡]  常态（不含 logits 尖峰）")
print(f"  模型权重          {p(weights_gb)} GB")
print(f"  LoRA 权重/梯度     {p(lora_gb + grad_gb)} GB")
print(f"  AdamW 动量(fp32)   {p(adam_gb)} GB")
print(f"  激活(梯度检查点)   {p(act_gb)} GB")
print(f"  小计 ≈ {p(steady)} GB")
print("[trainer 卡]  logits 尖峰（accelerate bf16→fp32 转换）")
print(f"  bf16 logits       {p(logits_bf16)} GB")
print(f"  fp32 转换          {p(logits_fp32)} GB")
print(f"  典型峰值 ≈ {p(peak_typ)} GB")
print(f"  最坏峰值 ≈ {p(peak_worst)} GB（含 log_softmax 第二份 fp32 副本）")
print()
print(f"结论: server 卡需要 ≥ {p(server_gb)} GB 空闲；")
print(f"      trainer 卡需要 ≥ {p(peak_typ)} GB 空闲（最坏 {p(peak_worst)} GB）。")
print(f"以 A100-80G 计，trainer 卡上的其他租户需 ≤ {p(gpu_total - peak_typ)} GB 才安全。")
EOF
