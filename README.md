<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.png">
    <img src="docs/assets/logo-light.png" width="440" alt="SieteRL">
  </picture>
</p>

<p align="center">EN | <a href="README_CN.md">中文</a></p>

<p align="center"><i>Reinforcement learning for software-engineering agents with verifiable rewards.</i></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-04648c.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/TRL-1.8.0-04648c.svg" alt="TRL 1.8.0">
  <a href="https://github.com/Sevendogs077/siete-rl/actions/workflows/ci.yml"><img src="https://github.com/Sevendogs077/siete-rl/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-6e7781.svg" alt="MIT License"></a>
</p>

SieteRL is a GRPO post-training project for software-engineering agents. The initial policy is SWE-Gym OpenHands-7B-Agent, an SFT checkpoint of Qwen2.5-Coder-7B. The agent fixes repository-level bugs through multi-turn interaction in isolated containers; each submitted patch is scored by the task's executable tests, and the verifier outcome is the reward. Trained policies are evaluated on SWE-bench Verified.

> [!NOTE]
> SieteRL is under active development. Project structure, APIs, and experimental configurations may change.

## Core components

**OpenHands-compatible scaffold** — an in-repo implementation of the three-tool interaction protocol used by SWE-Gym OpenHands-7B-Agent. The agent runs commands via `execute_bash`, browses and edits the repository via `str_replace_editor`, and submits its patch with `finish`, all inside isolated task containers. Training and SWE-bench Verified evaluation share the same prompts, tool parser, and multi-turn state machine.

**Liger Kernel** — training enables Liger Kernel through TRL by default. Its chunked loss never materializes the full logits tensor, reducing the memory footprint of 32K-context, 7B LoRA GRPO training; importance-sampling correction is applied per token, since sequence-level weights vanish on long agent trajectories.

## Results

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/results-chart-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/results-chart-light.png">
    <img src="docs/assets/results-chart-light.png" width="860" alt="SWE-bench Verified results">
  </picture>
</p>

Full results and run details: [docs/results.md](https://github.com/Sevendogs077/siete-rl/blob/main/docs/results.md).

## Reproduction

Training requires 2 GPUs; evaluation requires 1 GPU. Follow the [setup guide](docs/setup.md) to prepare the environment, datasets, base model, dedicated Docker daemon, task images, and evaluation harness.

```bash
uv sync
bash scripts/prepare.sh          # pull task images and generate assets
bash scripts/qualify.sh          # check config, data, images, model, and GPU
CUDA_VISIBLE_DEVICES=0,1 bash scripts/grpo.sh                 # start training
CUDA_VISIBLE_DEVICES=0 bash scripts/eval.sh outputs/<run-id>  # start evaluation
# override the script's 16/4 worker defaults on a smaller host
EVAL_ROLLOUT_WORKERS=8 EVAL_HARNESS_WORKERS=2 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval.sh outputs/<run-id>
```

`scripts/eval.sh` defaults rollout/harness workers to `16/4`; explicit environment
values override them. Direct `python -m siete_rl.eval` calls retain safe `1/1` defaults.
Each simultaneous evaluation process has its own worker pool, so size the combined
Docker CPU and memory limits across all running evaluations.

> [!WARNING]
> The agent executes model-generated code. Only connect a dedicated, non-production Docker daemon.

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

The code in this repository is released under the [MIT License](LICENSE).
