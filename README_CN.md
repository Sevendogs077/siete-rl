<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.png">
    <img src="docs/assets/logo-light.png" width="440" alt="SieteRL">
  </picture>
</p>

<p align="center"><a href="README.md">EN</a> | 中文</p>

<p align="center"><i>Reinforcement learning for software-engineering agents with verifiable rewards.</i></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-04648c.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/TRL-1.8.0-04648c.svg" alt="TRL 1.8.0">
  <a href="https://github.com/Sevendogs077/siete-rl/actions/workflows/ci.yml"><img src="https://github.com/Sevendogs077/siete-rl/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-6e7781.svg" alt="MIT License"></a>
</p>

SieteRL 是一个面向软件工程 Agent 的 GRPO 后训练项目。初始 policy 为 SWE-Gym OpenHands-7B-Agent（基于 Qwen2.5-Coder-7B 的 SFT checkpoint）。Agent 在隔离容器中进行多轮仓库级 bug 修复；提交的补丁由任务自带的可执行测试验证，verifier 结果即为奖励。训练后的 policy 在 SWE-bench Verified 上统一评测。

> [!NOTE]
> SieteRL 仍在积极开发中，项目结构、接口与实验配置可能变动。

## 项目核心

**OpenHands 兼容 scaffold** — 本仓库自行实现与 SWE-Gym OpenHands-7B-Agent 对齐的三工具交互协议。Agent 在隔离的任务容器中通过 `execute_bash` 运行命令、通过 `str_replace_editor` 浏览和修改仓库，并以 `finish` 提交当前 patch；训练与 SWE-bench Verified 评测共用同一套 prompt、工具解析器和多轮状态机。

**Liger Kernel** — 训练默认通过 TRL 启用 Liger Kernel：chunked loss 不保存完整 logits，降低 32K 上下文、7B LoRA GRPO 的显存开销。重要性采样校正按 token 级计算，因为长 Agent 轨迹上序列级权重会趋近于零。

## 实验结果

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/results-chart-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/results-chart-light.png">
    <img src="docs/assets/results-chart-light.png" width="860" alt="SWE-bench Verified results">
  </picture>
</p>

完整结果与运行记录见 [docs/results.md](https://github.com/Sevendogs077/siete-rl/blob/main/docs/results.md)。

## 复现运行

训练需要 2 张 GPU，评测需要 1 张。环境、数据、模型、专用 Docker daemon、任务镜像与评测 harness 的准备步骤见[环境与资产准备指南](docs/setup.md)。

```bash
uv sync
bash scripts/prepare.sh          # 拉取任务镜像并生成 assets
bash scripts/qualify.sh          # 检查配置、数据、镜像、模型与 GPU
CUDA_VISIBLE_DEVICES=0,1 bash scripts/grpo.sh                 # 启动训练
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh outputs/<run-id>  # 启动评测
```

> [!WARNING]
> Agent 会执行模型生成的代码，请仅连接专用的非生产 Docker daemon。

## 项目结构

```text
src/siete_rl/   训练、环境、验证、评测与记录
configs/        当前实验配置
scripts/        资产准备、资格检查、训练与评测入口
tests/          单元测试及需要显式启用的集成测试
docs/           实验文档与补充说明
```

## 参考与来源

<p align="center">
  <a href="https://github.com/huggingface/trl">TRL</a> ·
  <a href="https://github.com/SWE-Gym/SWE-Gym">SWE-Gym</a> ·
  <a href="https://github.com/SWE-bench/SWE-bench">SWE-bench</a> ·
  <a href="https://github.com/All-Hands-AI/OpenHands">OpenHands</a> ·
  <a href="https://github.com/linkedin/Liger-Kernel">Liger Kernel</a> ·
  <a href="https://huggingface.co/NovaSky-AI/SWE-Gym-OpenHands-7B-Agent">SWE-Gym OpenHands-7B-Agent</a> ·
  <a href="https://huggingface.co/datasets/SumanthRH/SWE-Gym-Subset">SumanthRH/SWE-Gym-Subset</a> ·
  <a href="https://github.com/NovaSky-AI/SkyRL">SkyRL</a>
</p>

## 开源许可

本仓库代码以 [MIT License](LICENSE) 发布。
