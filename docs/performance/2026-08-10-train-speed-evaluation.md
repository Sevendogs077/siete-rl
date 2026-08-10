# 训练加速评估设计与执行计划

## 目标

在 commit `4699d00973bc07b763ae25d210d27ead645b76bf` 上，用隔离 worktree 和 GPU 0/1 评估训练吞吐瓶颈。产出应是低风险、可复核的参数或局部实现建议，不修改训练语义，不影响主工作区。

## 约束与原则

- worktree 分支：`perf/train-speed-eval-4699d00`。实验 harness 可在该分支单独提交，但自动校验 `src/` 与 `configs/` 相对基准 commit 无差异，保证实际训练代码仍精确对应 `4699d00`。
- 复用主仓库已锁定的 `.venv`、模型、数据和任务资产，只读访问；实验输出写入 worktree 的 `benchmark_results/`。
- 真实训练固定任务、seed、模型、采样参数和 Docker 配额，一次只改变一个变量。
- 单次真实训练最多 1 个 optimizer step；先小组筛选，再只对有收益的候选做生产规模确认。
- 使用 A/B/A 顺序控制首次加载、文件缓存和系统抖动；效果低于 5% 视为噪声区，不据此改默认值。
- 正确性护栏：退出码、run 状态、global step、infra error、模型更新、轨迹数、completion token、tool call、reward 分布和输出文件契约。
- 不测试会明显改变目标函数或采样分布的“加速”，例如缩短 completion 上限、减少正式 `num_generations`、跳过 verifier。

## 评估层次

### 1. 静态与历史基线

读取当前配置、训练/launcher/Docker/reward/recorder 路径，以及两次已完成 100-step run。分清初始化开销、steady step、生成、工具、verifier、反向和持久化，避免用旧版本的绝对时间代替当前结论。

### 2. CPU/Docker 微基准

对固定任务执行真实 `SWEEnvironment.reset`，比较 16 个环境串行与有界并发；同一批已打开环境上比较工具调用串行与并发。每个场景至少 3 次，报告中位数和范围。微基准只用于决定是否值得改动 reset/线程池，不外推为完整 step 加速。

### 3. 双 GPU 短训练

- GPU 0：vLLM server；GPU 1：LoRA trainer。
- 筛选规模：`num_generations=4`、`generation_batch_size=4`、`gradient_accumulation_steps=4`、`max_steps=1`。
- 生产确认：保持正式 16/16/16，仅 `max_steps=1`。
- 第一候选：vLLM `gpu_memory_utilization` 从 0.45 提到 0.60；其余完全一致。
- baseline 与候选按 A/B/A 跑；候选需同时满足无 OOM/重启、正确性护栏通过且相对两次 A 的中位数至少快 5%，才进入生产规模 A/B 确认。
- 若 baseline 的 trainer 峰值显存有至少 20 GiB 余量，再评估关闭 gradient checkpointing；否则直接判定风险大于收益，不运行。
- `tool_parallel_workers` / `verifier_parallel_workers` 先由微基准筛选。只有 8 workers 比 16 workers 稳定快至少 5% 时，才进入真实训练确认。

## 自动化产物

- `scripts/train_speed_benchmark.py`：生成单变量配置、校验 commit/cleanliness、顺序启动训练、每 2 秒采集 GPU 0/1 指标、写 manifest 和每臂 summary。训练仍以 `save_steps=1` 产生受运行时契约要求的 checkpoint，不能用 `save_strategy=no` 跳过正确性检查。
- `scripts/docker_parallel_benchmark.py`：真实 Docker reset/tool 微基准，保证异常时关闭全部环境。
- `scripts/summarize_train_speed.py`：从 run JSON、metrics JSONL、GPU CSV 和 wall time 生成统一结果。
- `tests/unit/test_train_speed_benchmark.py`：先验证配置单变量、A/B/A 顺序、阈值判定和结果解析，再实现脚本。

## 执行顺序

1. 记录 worktree、commit、Python/torch/TRL/vLLM、GPU、Docker 和基线测试。
2. 先写自动化测试并确认失败，再实现纯实验脚本，使测试转绿。
3. 跑 Docker 微基准 3 次；不足 5% 的差异不升级到真实训练。
4. 跑 4-generation A/B/A；检查完整性与 GPU 峰值。
5. 仅对筛选通过项跑 16-generation A/B；如两层方向不一致，以生产规模为准。
6. 汇总“立即调整 / 需局部改动验证 / 不建议”的方向，并保留所有原始结果与复现命令。

## 停止条件

- 任一 arm 出现 OOM、vLLM 重启、Docker 基础设施错误或训练契约失败：该候选立即停止，不通过调低正确性要求掩盖。
- 单次 wall time 超过 30 分钟：终止该 arm，记录为不适合短评估；不自动重复。
- 结论对任务或运行顺序敏感、相对收益不足 5%：结论记为不确定，不改默认配置。

## 已完成结果：Docker/CPU 微基准

执行时间：2026-08-10。固定任务 `getmoto__moto-5189`，每批 16 个真实 `SWEEnvironment`，每个 worker 设置 3 次，使用轮换顺序控制 daemon 热身。原始机器可读结果保存在 worktree 的 `benchmark_results/docker_parallel.json`（被 gitignore，不进入产品树）。

| 阶段 | workers=1 中位数（范围） | workers=8 中位数（范围） | workers=16 中位数（范围） |
| --- | ---: | ---: | ---: |
| environment reset | 31.61 s（31.46–31.70） | 12.54 s（12.25–12.57） | 9.47 s（9.44–10.07） |
| 16 个轻量真实工具调用 | 4.94 s（4.89–5.07） | 0.65 s（0.64–0.72） | 0.38 s（0.35–0.41） |

结论：

- 当前 `tool_parallel_workers=16` 有充分证据，应保留；降到 8 会使该微基准的工具 wall time 增加约 71%，不值得进入昂贵的完整训练 A/B。
- 当前 TRL 路径逐个调用 `environment.reset`。若将 16 个互相隔离的 reset 做有界并发，微基准从 31.61 s 降到 9.47 s（约 70% 的 reset 阶段缩短，单个 step 的绝对上限约 22 s）。这是目前最坚实的局部代码候选，但不能把 22 s 直接等同于完整 step 收益。
- 8→16 的 reset 中位数仍改善约 24.5%，且三次范围没有显示严重抖动；在本机 80 CPU、独立 Docker daemon 上暂未看到 16 workers 过饱和。
- 最终兜底清扫返回空列表，说明正常关闭路径已清理全部 benchmark 容器。

下一阶段按用户要求暂停，尚未启动 GPU A/B。恢复时从 `scripts/train_speed_benchmark.py --phase screen` 开始。
