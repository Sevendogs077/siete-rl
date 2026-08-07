<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.png">
    <img src="docs/assets/logo-light.png" width="440" alt="SieteRL">
  </picture>
</p>

<p align="center"><a href="README.md">中文</a> | EN</p>

<p align="center"><i>Reinforcement learning for verifiable software-engineering agents.</i></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://github.com/Sevendogs077/siete-rl/actions/workflows/ci.yml"><img src="https://github.com/Sevendogs077/siete-rl/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/TRL-1.8.0-orange.svg" alt="TRL 1.8.0">
  <a href="https://github.com/SWE-Gym/SWE-Gym"><img src="https://img.shields.io/badge/training-SWE--Gym-2ea44f.svg" alt="Training on SWE-Gym"></a>
  <a href="https://github.com/SWE-bench/SWE-bench"><img src="https://img.shields.io/badge/evaluation-SWE--bench%20Verified-6f42c1.svg" alt="Evaluated on SWE-bench Verified"></a>
</p>

SieteRL trains software-engineering agents to fix real bugs with GRPO reinforcement learning.

Starting from Qwen2.5-Coder-7B (the SWE-Gym OpenHands-7B-Agent SFT policy), the agent repairs SWE-Gym tasks through multi-turn interaction in isolated containers, and real test results directly become the reward. The trained policy is evaluated on SWE-bench Verified.

## Core components

**OpenHands scaffold** — a three-tool interaction protocol matching SWE-Gym OpenHands-7B-Agent. The agent runs commands via `execute_bash`, browses and edits the repository via `str_replace_editor`, and submits the current patch with `finish`, all inside isolated task containers. Training and SWE-bench Verified evaluation share the same prompts, tool parser, and multi-turn state machine.

**Liger Kernel** — training enables Liger Kernel through TRL by default, using a chunked loss to reduce the memory footprint of backpropagation for 32K-context, 7B LoRA GRPO. The configuration pins the compatible token-level importance sampling, and uses eager mode to avoid a known conflict between Torch 2.11 and Liger 0.8 on dynamic-shape recompilation.

## Results

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/results-chart-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/results-chart-light.png">
    <img src="docs/assets/results-chart-light.png" width="860" alt="SWE-bench Verified results">
  </picture>
</p>

Outcome distribution on SWE-bench Verified (500 tasks, one rollout per task). Full experiment records: [docs/experiment_log.md](docs/experiment_log.md).

## Reproduction

Training requires 2 GPUs; evaluation requires 1 GPU. See [docs/asset_preparation.md](docs/asset_preparation.md) for the preparation of datasets, the base model, the dedicated Docker daemon, task images, and the evaluation harness.

```bash
uv sync
bash scripts/prepare.sh          # pull task images and generate assets
bash scripts/qualify.sh          # check config, data, images, model, and GPU
CUDA_VISIBLE_DEVICES=0,1 bash scripts/grpo.sh                 # start training
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh outputs/<run-id>  # requires EVAL_HARNESS_PYTHON
```

> **Safety note:** the agent executes model-generated code. Only connect a dedicated, non-production Docker daemon.

## Project layout

```text
src/siete_rl/   training, environment, verification, evaluation, and run recording
configs/        current experiment configurations
scripts/        entry points for asset preparation, qualification, training, and evaluation
tests/          unit tests and explicitly enabled integration tests
docs/           experiment documentation and notes
```

## References

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

## License

The code in this repository is released under the [MIT License](LICENSE). Third-party models, datasets, and benchmark assets are subject to their own licenses.
