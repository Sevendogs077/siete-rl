<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.png">
    <img src="docs/assets/logo-light.png" width="440" alt="SieteRL">
  </picture>
</p>

<p align="center"><i>Reinforcement learning for verifiable software-engineering agents.</i></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/TRL-1.8.0-orange.svg" alt="TRL 1.8.0">
  <a href="https://github.com/SWE-Gym/SWE-Gym"><img src="https://img.shields.io/badge/training-SWE--Gym-2ea44f.svg" alt="Training on SWE-Gym"></a>
  <a href="https://github.com/SWE-bench/SWE-bench"><img src="https://img.shields.io/badge/evaluation-SWE--bench%20Verified-6f42c1.svg" alt="Evaluated on SWE-bench Verified"></a>
</p>

SieteRL 面向可验证的软件工程 Agent 强化学习，以 Qwen2.5-Coder-7B 为模型基座，并从 SWE-Gym OpenHands-7B-Agent 的 SFT 策略出发。Agent 通过由 `execute_bash`、`str_replace_editor` 与 `finish` 构成的 OpenHands scaffold，在 SWE-Gym 的可执行任务环境中生成多轮修复轨迹；提交的补丁随后进入隔离容器接受真实测试，测试结果被转换为 outcome reward，并由 TRL 的 `GRPOTrainer` 用于 LoRA 策略优化。训练所得策略最终在 SWE-bench Verified 上评估。

## 项目核心

**OpenHands scaffold** — 实现与 SWE-Gym OpenHands-7B-Agent 相匹配的三工具交互协议。Agent 在隔离的任务容器中通过 `execute_bash` 运行命令、通过 `str_replace_editor` 浏览和修改仓库，并以 `finish` 提交当前 patch；训练与 SWE-bench Verified 评测共用同一套 prompt、工具解析器和多轮状态机。

**Liger Kernel** — 训练默认通过 TRL 启用 Liger Kernel，以 chunked loss 降低 32K 上下文、7B LoRA GRPO 反向传播的显存开销。当前配置固定使用兼容的 token-level importance sampling，并以 eager mode 规避 Torch 2.11 与 Liger 0.8 在动态 shape 重编译时的已知冲突。

## 实验结果

| Run | Resolved | Empty Patch | Context Limit |
|:---:|:---:|:---:|:---:|
| GRPO | **56 / 500（11.2%）** | 152 / 500（30.4%） | 179 / 500（35.8%） |

<p>
  完整实验记录见
  <a href="docs/experiment_log.md"><img src="https://img.shields.io/badge/docs-experiment_log-4b5563.svg?logo=readthedocs&logoColor=white" alt="Experiment log"></a>
</p>

## 复现运行

开始前需额外准备 SWE-Gym parquet 数据与任务镜像、[SWE-Gym OpenHands-7B-Agent](https://huggingface.co/NovaSky-AI/SWE-Gym-OpenHands-7B-Agent) 模型及 tokenizer，以及固定版本的 SWE-bench harness。

```bash
uv sync
bash scripts/prepare.sh          # 数据就绪后拉取任务镜像并生成 assets
bash scripts/qualify.sh          # 检查配置、数据、镜像、模型与 GPU
bash scripts/grpo.sh             # 启动训练
bash scripts/eval.sh outputs/<run-id>
```

> **安全提示：** Agent 会执行模型生成的代码，请仅连接专用的非生产 Docker daemon。

## 项目结构

```text
src/siete_rl/   训练、环境、验证、评测与运行记录
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

本仓库代码以 [MIT License](LICENSE) 发布；第三方模型、数据集与 benchmark assets 适用各自的许可条款。
