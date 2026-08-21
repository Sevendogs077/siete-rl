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

SieteRL 使用强化学习训练软件工程 Agent。项目以 SWE-Gym OpenHands-7B-Agent 为起点，Agent 在隔离容器中通过多轮交互解决仓库级任务，训练结果统一在 SWE-bench Verified 上评测。

> [!NOTE]
> SieteRL 仍在积极开发中，项目结构、接口与实验配置可能变动。

## 实验结果

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/results-chart-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/results-chart-light.png">
    <img src="docs/assets/results-chart-light.png" width="860" alt="SWE-bench Verified results">
  </picture>
</p>

完整结果与运行记录见 [docs/results.md](docs/results.md)。

## Details

- **Layered Rewards** — 完全修复得 `1.0`；未完全修复的补丁仅在成功应用且全部 P2P 测试通过时获得 `0.20p²`，其中 `p` 为 F2P 通过率。
- **全零组校准** — 当组内所有 rollout 均失败时，固定的 `0.1` 参考奖励仍能保留训练信号。
- **Sign-aware Process Mask** — 正优势下屏蔽格式错误及第 3 次起完全相同的 Bash/editor 动作；负优势下保留这些 token，让错误步骤接受惩罚而不是分享正奖励。
- **Failure-aware Rollout Handling** — 基础设施失败不进入组内均值，也不产生梯度；物理截断仅屏蔽最后一个不完整 turn，其余异常结束的最终补丁仍会参与验证。
- **Dr. GRPO Objective** — 使用全局最大生成长度归一化 loss，并结合 `0.16/0.24` 非对称裁剪与 token 级 vLLM 重要性修正训练 32K 工具调用轨迹。

## 复现运行

默认使用 4 张 GPU 训练、GPU 0 评测，双卡配置及环境准备见[环境与资产准备指南](docs/setup.md)。

```bash
uv sync
bash scripts/prepare.sh
bash scripts/qualify.sh
bash scripts/stage1.sh
bash scripts/stage2.sh outputs/<stage1-run-id>
bash scripts/eval.sh outputs/<stage2-run-id>
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
  <a href="https://github.com/linkedin/Liger-Kernel">Liger Kernel</a> ·
  <a href="https://github.com/OpenHands/OpenHands">OpenHands</a> ·
  <a href="https://github.com/NovaSky-AI/SkyRL">SkyRL</a> ·
  <a href="https://huggingface.co/datasets/SumanthRH/SWE-Gym">SumanthRH/SWE-Gym</a> ·
  <a href="https://github.com/SWE-bench/SWE-bench">SWE-bench</a> ·
  <a href="https://github.com/SWE-Gym/SWE-Gym">SWE-Gym</a> ·
  <a href="https://huggingface.co/NovaSky-AI/SWE-Gym-OpenHands-7B-Agent">SWE-Gym OpenHands-7B-Agent</a> ·
  <a href="https://github.com/huggingface/trl">TRL</a>
</p>

## 开源许可

本仓库代码以 [MIT License](LICENSE) 发布。
