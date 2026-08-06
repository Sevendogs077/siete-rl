# 分层 Outcome Reward（指数型部分分）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把训练奖励从纯 0/1 改为分层 outcome reward：apply 失败/未提交为 0，resolved 为 1，中间态按 `R = f(p)·e^(−μk)` 给指数型部分分（p = FAIL_TO_PASS 通过比例，k = PASS_TO_PASS 失败个数）。

**Architecture:** 灵感来自《Signal Reshaping for GRPO in Weak-Feedback Agentic Code Repair》的贡献点 1（分层结果奖励），但本项目有 pytest test oracle（强反馈），**不需要 LLM judge**——用 verifier 真实测试结果替代语义判定。核心是一个纯函数评分模块 `scoring.py`（易单测、易消融），接线点只有三处：`Evaluation` 模型携带测试清单、`SWEEnvironment._finalize` 按配置选择评分路径、yaml 配置开关。GRPO 目标函数、advantage 计算、TRL 镜像代码一律不动。

**Tech Stack:** Python 3.12 / pydantic StrictModel / TRL 1.8.0 GRPO / pytest。

**设计约束（论文贡献点 4，所有 Task 必须遵守）：**
- 不改 GRPO 目标函数与 advantage 计算；只重塑 reward 输入信号。
- 中间分只来自 verifier 真实测试事实，不引入任何模型自评/judge。

**不做清单（论文贡献点 5 的负结果 + 范围裁剪，本计划明确不做）：**
- 不做 token 级 KL 蒸馏惩罚写进 reward（论文实测无可用工作区间）。
- 不做步骤级过程分数 loss 权重（需改 TRL `_compute_loss`，后续单独立项评估）。
- 不做 rollout 治理的失败原因路由（`infra_error` 仍原样 raise，后续单独立项）。

**奖励公式（已与需求方确认，λ=8，μ=ln2）：**

| 情况 | R |
|---|---|
| 未提交 / 空 patch / `check_failed` / `apply_failed` / pytest 未启动 | 0.0 |
| applied 且 pytest 运行，FAIL_TO_PASS 过比例 p、PASS_TO_PASS 挂 k 个 | `expm1(λp)/expm1(λ) · exp(−μk)` |
| `resolved` | 1.0（= f(1)·e⁰，公式在满分处无跳变） |

参考值（λ=8，μ=ln2）：p=2/3≈0.018，p=0.8≈0.20，p=0.8 且 k=1≈0.10。

---

### Task 1: 纯函数评分模块 `scoring.py`

**Files:**
- Create: `src/swe_agent/scoring.py`
- Test: `tests/unit/test_scoring.py`

- [ ] **Step 1: Write the failing test**

创建 `tests/unit/test_scoring.py`：

```python
"""分层 outcome reward 评分纯函数测试。"""

import math

import pytest

from swe_agent.models import Verification
from swe_agent.scoring import layered_score, parse_pytest_summary

LAMBDA = 8.0
MU = math.log(2)


def _verification(
    result: str = "unresolved",
    apply_status: str = "applied",
    exit_code: int = 1,
    stdout: str = "",
) -> Verification:
    return Verification.model_validate(
        {
            "result": result,
            "patch_apply_status": apply_status,
            "pytest_started": apply_status == "applied",
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": "",
        }
    )


PYTEST_STDOUT = """
=================== short test summary info ====================
PASSED mypy/test/teststubgen.py::TestStubgenPythonSuite::testObjectBaseClass
PASSED mypy/test/teststubgen.py::TestStubgenPythonSuite::testGenericClassTypeVarTuple
FAILED mypy/test/teststubgen.py::TestStubgenPythonSuite::testGenericClassTypeVarTuple_semanal - AssertionError
FAILED mypy/test/teststubgen.py::TestStubgenPythonSuite::testOtherRegressed - RuntimeError
"""


def test_parse_pytest_summary_splits_passed_and_failed():
    passed, failed = parse_pytest_summary(PYTEST_STDOUT)
    assert "mypy/test/teststubgen.py::TestStubgenPythonSuite::testObjectBaseClass" in passed
    assert "mypy/test/teststubgen.py::TestStubgenPythonSuite::testGenericClassTypeVarTuple" in passed
    assert any("testGenericClassTypeVarTuple_semanal" in n for n in failed)
    assert any("testOtherRegressed" in n for n in failed)
    # FAILED 行尾的 " - reason" 不得进入 nodeid
    assert all(" - " not in n for n in failed)


def test_resolved_is_exactly_one():
    v = _verification(result="resolved", apply_status="applied", exit_code=0)
    assert layered_score(
        verification=v, fail_to_pass=["t1"], pass_to_pass=["t2"],
        lambda_=LAMBDA, mu=MU,
    ) == 1.0


def test_apply_failed_is_zero():
    v = _verification(apply_status="apply_failed")
    assert layered_score(
        verification=v, fail_to_pass=["t1"], pass_to_pass=[],
        lambda_=LAMBDA, mu=MU,
    ) == 0.0


def test_check_failed_is_zero():
    v = _verification(apply_status="check_failed")
    assert layered_score(
        verification=v, fail_to_pass=["t1"], pass_to_pass=[],
        lambda_=LAMBDA, mu=MU,
    ) == 0.0


def test_partial_ratio_uses_exponential_shape():
    # F2P 两个测试过一个：p=0.5，λ=8 → expm1(4)/expm1(8) ≈ 0.018
    score = layered_score(
        verification=_verification(stdout=PYTEST_STDOUT),
        fail_to_pass=["testGenericClassTypeVarTuple", "testGenericClassTypeVarTuple_semanal"],
        pass_to_pass=[],
        lambda_=LAMBDA, mu=MU,
    )
    assert score == pytest.approx(math.expm1(4.0) / math.expm1(8.0), rel=1e-9)
    assert score < 0.02  # 线性会给 0.5，指数型必须压到接近 0


def test_regression_halves_score_per_failed_p2p():
    # p=0.5 且 PASS_TO_PASS 挂 1 个（testOtherRegressed 不在 F2P 里 → 视为回归）
    score = layered_score(
        verification=_verification(stdout=PYTEST_STDOUT),
        fail_to_pass=["testGenericClassTypeVarTuple", "testGenericClassTypeVarTuple_semanal"],
        pass_to_pass=["testOtherRegressed"],
        lambda_=LAMBDA, mu=MU,
    )
    base = math.expm1(4.0) / math.expm1(8.0)
    assert score == pytest.approx(base * 0.5, rel=1e-9)


def test_all_f2p_passed_but_unresolved_is_near_one_not_one():
    # F2P 全过但有 P2P 回归 → result 是 unresolved；p=1, k=1 → 0.5
    stdout = """
PASSED a.py::test_target
FAILED a.py::test_regressed - RuntimeError
"""
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["test_target"],
        pass_to_pass=["test_regressed"],
        lambda_=LAMBDA, mu=MU,
    )
    assert score == pytest.approx(0.5, rel=1e-9)


def test_empty_fail_to_pass_treated_as_full_ratio():
    score = layered_score(
        verification=_verification(stdout="PASSED a.py::test_x\n"),
        fail_to_pass=[],
        pass_to_pass=[],
        lambda_=LAMBDA, mu=MU,
    )
    assert score == pytest.approx(1.0, rel=1e-9)


def test_parametrized_and_dotted_names_match_by_last_segment():
    # F2P 清单可能给完整 nodeid、裸测试名或带参数化后缀
    stdout = "PASSED tests/test_a.py::TestSuite::test_case[param]\n"
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["test_case[param]"],
        pass_to_pass=[],
        lambda_=LAMBDA, mu=MU,
    )
    assert score == pytest.approx(1.0, rel=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_scoring.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'swe_agent.scoring'`

- [ ] **Step 3: Write minimal implementation**

创建 `src/swe_agent/scoring.py`：

```python
"""分层 outcome reward：指数型部分分纯函数。

设计来源：《Signal Reshaping for GRPO in Weak-Feedback Agentic Code Repair》
贡献点 1（分层结果奖励）的 test-oracle 改造版——本项目 verifier 直接运行
pytest，用真实测试结果替代论文中的 LLM judge 语义判定。

公式：R = f(p) · e^(−μk)，f(p) = (e^(λp) − 1)/(e^λ − 1)
  p = FAIL_TO_PASS 通过比例；k = PASS_TO_PASS 失败个数（不是比例——
  P2P 通常几十个，比例制会让单个回归几乎不扣分，门就失效了）。
  resolved 时 p=1、k=0，R = 1，公式在满分处无跳变。
只重塑 reward 输入信号；GRPO 目标函数与 advantage 计算不变（论文贡献点 4）。
"""

from __future__ import annotations

import math
import re

from swe_agent.models import Verification

_SUMMARY_LINE = re.compile(r"(?m)^(PASSED|FAILED|ERROR)\s+(\S+)")


def parse_pytest_summary(stdout: str) -> tuple[set[str], set[str]]:
    """解析 `pytest -rA` short summary，返回 (passed nodeids, failed nodeids)。

    FAILED 行可能带 " - reason" 后缀，`\\S+` 在空格处截断，不会带入。
    ERROR 计入 failed（收集/fixture 错误等价于测试失败）。
    """

    passed: set[str] = set()
    failed: set[str] = set()
    for status, nodeid in _SUMMARY_LINE.findall(stdout):
        if status == "PASSED":
            passed.add(nodeid)
        else:
            failed.add(nodeid)
    return passed, failed


def _normalize_test_name(test_id: str) -> str:
    """归一到裸测试名：取 nodeid 最后一段，保留参数化后缀。"""

    return test_id.split("::")[-1].strip()


def layered_score(
    *,
    verification: Verification,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    lambda_: float,
    mu: float,
) -> float:
    """按分层公式给 applied-but-unresolved 的 patch 打部分分。"""

    if verification.result == "resolved":
        return 1.0
    if verification.patch_apply_status != "applied" or not verification.pytest_started:
        return 0.0
    passed, failed = parse_pytest_summary(verification.stdout)
    passed_names = {_normalize_test_name(n) for n in passed}
    failed_names = {_normalize_test_name(n) for n in failed}
    f2p = {_normalize_test_name(t) for t in fail_to_pass}
    p2p = {_normalize_test_name(t) for t in pass_to_pass}
    p = len(f2p & passed_names) / len(f2p) if f2p else 1.0
    k = len(p2p & failed_names)
    return math.expm1(lambda_ * p) / math.expm1(lambda_) * math.exp(-mu * k)
```

注意：`math.expm1` 防大 λ 数值溢出，不要用 `exp(x) - 1`。

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_scoring.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/swe_agent/scoring.py tests/unit/test_scoring.py
git commit -m "feat: add layered outcome reward scoring functions"
```

---

### Task 2: `Evaluation` 模型携带 FAIL_TO_PASS / PASS_TO_PASS 清单

评分需要知道哪些测试是 F2P、哪些是 P2P。这两个清单在 official parquet 行里（`swegym.py:25-34` 的 `COMPARE_FIELDS` 已含），是 JSON 编码的字符串列表，目前被丢弃。

**Files:**
- Modify: `src/swe_agent/models.py:46-49`（`Evaluation` 模型）
- Modify: `src/swe_agent/swegym.py:69-110`（`load_task_instance`）
- Test: `tests/unit/test_swegym.py`

- [ ] **Step 1: Write the failing test**

在 `tests/unit/test_swegym.py` 追加（参照该文件既有 fixture 风格构造 parquet/资产，若已有 `load_task_instance` 测试则在其 fixture 上扩展）：

```python
def test_load_task_instance_carries_test_lists(tmp_path, monkeypatch):
    # 复用本文件既有 fixture 构造单任务资产后：
    sample, evaluation = load_task_instance(config, project_root, "python__mypy-16869")
    assert evaluation.fail_to_pass == ["testGenericClassTypeVarTuple"]
    assert evaluation.pass_to_pass == ["testObjectBaseClass"]


def test_load_task_instance_rejects_non_list_test_field(tmp_path, monkeypatch):
    # official 行的 FAIL_TO_PASS 不是 JSON 字符串列表时：
    with pytest.raises(SWEGymContractError, match="FAIL_TO_PASS"):
        load_task_instance(config, project_root, "python__mypy-16869")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_swegym.py -v -k test_lists`
Expected: FAIL，`TypeError: Evaluation.__init__() got an unexpected keyword argument 'fail_to_pass'`

- [ ] **Step 3: Write minimal implementation**

`src/swe_agent/models.py`，扩展 `Evaluation`：

```python
class Evaluation(StrictModel):
    """只在进程内交给 verifier 的私有运行时事实。"""

    offline_eval_script: str = Field(min_length=1)
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
```

`src/swe_agent/swegym.py`，在 `load_task_instance` 的 return 前解析两个清单（`json` 已在文件头导入）：

```python
def _load_test_list(row: dict[str, Any], field: str, task_id: str) -> list[str]:
    raw = row[field]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SWEGymContractError(f"{task_id}: {field} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list) or not all(isinstance(t, str) and t for t in raw):
        raise SWEGymContractError(f"{task_id}: {field} must be a JSON list of test ids")
    return list(raw)
```

并把 `load_task_instance` 的 return 改为：

```python
    return Sample(task=task, environment=environment), Evaluation(
        offline_eval_script=offline,
        fail_to_pass=_load_test_list(official, "FAIL_TO_PASS", task_id),
        pass_to_pass=_load_test_list(official, "PASS_TO_PASS", task_id),
    )
```

注意：`Evaluation` 其他构造点（测试 fixture、`tests/test_eval.py:151` 附近）会因 StrictModel 需要补字段或靠默认值通过——跑全量单测确认。

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit/test_swegym.py tests/unit/test_models.py tests/test_eval.py -v`
Expected: 全部 PASS（若有 fixture 构造 `Evaluation(...)` 失败，补上两个清单字段）

- [ ] **Step 5: Commit**

```bash
git add src/swe_agent/models.py src/swe_agent/swegym.py tests/
git commit -m "feat: carry FAIL_TO_PASS/PASS_TO_PASS lists in Evaluation"
```

---

### Task 3: 环境 finalize 接线 + 配置开关

**Files:**
- Modify: `src/swe_agent/config.py:101`（`GRPOConfigValues`）
- Modify: `src/swe_agent/environment.py`（`__init__` 与 `_finalize`，约 193-239 行）
- Modify: `src/swe_agent/train.py`（`environment_factory` 约 491-500 行、`_recording_reward` 约 695-759 行、`_native_policy_path_reached` 约 1396-1413 行）
- Test: `tests/unit/test_environment.py`、`tests/unit/test_config.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_config.py` 追加：

```python
def test_grpo_config_accepts_layered_reward_type():
    values = _minimal_grpo_values(reward_type="layered")  # 参照本文件既有 helper
    assert values.reward_type == "layered"
    assert values.layered_lambda == 8.0
    assert values.layered_mu == pytest.approx(math.log(2))


def test_grpo_config_rejects_nonpositive_lambda():
    with pytest.raises(ValidationError):
        _minimal_grpo_values(reward_type="layered", layered_lambda=0.0)
```

`tests/unit/test_environment.py` 追加（参照该文件既有的 submitted + verifier stub 测试，把 verifier stub 的返回换成 applied/unresolved 且 stdout 带 `-rA` 摘要）：

```python
def test_finalize_layered_gives_partial_score():
    # 构造 submitted 环境，verifier 返回 unresolved + 部分通过的 pytest 输出
    env = _make_submitted_env(reward_type="layered")  # 参照既有 helper
    reward = env._finalize(completion=None)
    assert 0.0 < reward < 1.0


def test_finalize_binary_unchanged_by_default():
    env = _make_submitted_env()  # 默认 reward_type="binary_verifier"
    reward = env._finalize(completion=None)
    assert reward in (0.0, 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_environment.py -v -k "layered or binary_unchanged"`
Expected: FAIL（`reward_type` Literal 拒绝 `"layered"`；`SWEEnvironment.__init__` 无 `reward_type` 参数）

- [ ] **Step 3: Write minimal implementation**

`src/swe_agent/config.py` 的 `GRPOConfigValues`：

```python
    reward_type: Literal["binary_verifier", "layered"]
    layered_lambda: float = Field(default=8.0, gt=0.0)
    layered_mu: float = Field(default=0.6931471805599453, gt=0.0)  # ln2：每个 P2P 回归减半
```

`src/swe_agent/environment.py` 的 `SWEEnvironment.__init__` 增加三个关键字参数（默认值保证既有调用与测试不破）：

```python
        reward_type: str = "binary_verifier",
        layered_lambda: float = 8.0,
        layered_mu: float = 0.6931471805599453,
```

存入同名私有属性。`_finalize` 末尾（现 `self._reward = 1.0 if ... == "resolved" else 0.0` 一行）改为：

```python
        if self._reward_type == "layered":
            self._reward = layered_score(
                verification=self._verification,
                fail_to_pass=self._evaluation.fail_to_pass,
                pass_to_pass=self._evaluation.pass_to_pass,
                lambda_=self._layered_lambda,
                mu=self._layered_mu,
            )
        else:
            self._reward = 1.0 if self._verification.result == "resolved" else 0.0
```

（文件头加 `from swe_agent.scoring import layered_score`。）

`src/swe_agent/train.py` 三处：

1. `environment_factory`（约 491-500 行）构造时传入：

```python
            environment = SWEEnvironment(
                task_context=task_context,
                sandbox_factory=sandbox_factory,
                verifier_factory=verifier_factory,
                output_limit_chars=config.chat.max_observation_chars,
                max_timeout_sec=config.docker.exec_timeout_sec,
                reward_type=config.grpo.reward_type,
                layered_lambda=config.grpo.layered_lambda,
                layered_mu=config.grpo.layered_mu,
            )
```

2. `_recording_reward(recorder, binary_reward)` 调用处（约 502 行）保持传入 `binary_reward` adapter（layered 的差异全在 `environment._finalize` 内部，adapter 不用改）；把 `_recording_reward` 末尾的 `reward.__name__ = "binary_reward"` 改为按实际类型命名——给 `_recording_reward` 加第三个参数 `reward_type: str = "binary_verifier"`，末尾 `reward.__name__ = f"{reward_type}_reward"`，调用处传 `config.grpo.reward_type`。这样 TRL 的 `rewards/<name>` 指标列名与配置一致。

3. `_native_policy_path_reached`（约 1408 行）把 `reward in (0.0, 1.0)` 改为 `isinstance(reward, float) and 0.0 <= reward <= 1.0`——layered 下中间分也是合法 reward。

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/unit -v`
Expected: 全部 PASS。特别注意 `tests/integration/test_trl_interfaces.py` 的 TRL 镜像守护测试不应受影响（本计划未动 `trainer.py`）；若 `test_environment.py` 有其他直接构造 `SWEEnvironment(...)` 的用例失败，按新签名补默认参数。

- [ ] **Step 5: Commit**

```bash
git add src/swe_agent/config.py src/swe_agent/environment.py src/swe_agent/train.py tests/
git commit -m "feat: wire layered reward through config and environment finalize"
```

---

### Task 4: 配置启用 + 端到端验证

**Files:**
- Modify: `configs/grpo_swegym_openhands_7b_lora.yaml`（`grpo` 段，约 59 行附近）
- Modify: `docs/exp_results.md`（记录本次消融设置与结果）

- [ ] **Step 1: 更新 yaml**

`configs/grpo_swegym_openhands_7b_lora.yaml` 的 `grpo` 段：

```yaml
  reward_type: layered
  layered_lambda: 8.0
  layered_mu: 0.6931471805599453
```

`configs/dapo_swegym_openhands_7b_lora.yaml` 暂不动（保持 binary 作为对照臂）。

- [ ] **Step 2: 配置加载验证**

Run: `uv run python -c "from swe_agent.config import load_project_config; c = load_project_config('configs/grpo_swegym_openhands_7b_lora.yaml'); print(c.grpo.reward_type, c.grpo.layered_lambda, c.grpo.layered_mu)"`
Expected: `layered 8.0 0.6931471805599453`

- [ ] **Step 3: 干跑验证**

Run: `bash scripts/dry_run.sh`（若该脚本走完整 rollout + reward 路径）
Expected: 正常结束；`outputs/<run-id>/` 的录制文件里出现 0 与 1 之外的中间 reward 值（若该 dry run 有样本 applied 但未 resolved）。检查方式：`grep -o '"reward": [0-9.]*' outputs/<最新run>/rollouts/*.json | sort -u`

- [ ] **Step 4: 记录消融设置**

在 `docs/exp_results.md` 追加一条记录：本 run 使用 `reward_type: layered, λ=8, μ=ln2`，对照臂为同配置 `binary_verifier`；观测指标为 `rewards/layered_reward` 均值、全 0 组占比、resolved 率。训练结果出来后回填数字。

- [ ] **Step 5: Commit**

```bash
git add configs/grpo_swegym_openhands_7b_lora.yaml docs/exp_results.md
git commit -m "exp: enable layered reward (lambda=8, mu=ln2) for grpo arm"
```

---

## Self-Review 记录

- **Spec 覆盖**：论文贡献点 1（分层奖励）→ Task 1-4 全覆盖，judge 替换为 test-oracle 是需求方确认的改造；贡献点 4（不动 GRPO 目标）→ 设计约束 + 全程未触 `trainer.py`；贡献点 5 → 不做清单。贡献点 2/3 经需求方决策移出范围，已注明后续单独立项。
- **类型一致性**：`layered_score` 的签名（`verification/fail_to_pass/pass_to_pass/lambda_/mu`）在 Task 1 定义、Task 3 调用一致；`Evaluation.fail_to_pass/pass_to_pass` 在 Task 2 定义、Task 3 使用一致；`reward_type/layered_lambda/layered_mu` 在 config、environment、train、yaml 四处拼写一致。
- **占位符**：无 TBD/TODO；Task 2 Step 1 的测试 fixture 标注了"参照既有 fixture"，是因为 `test_swegym.py` 的 parquet 构造细节需执行者现场对齐——这是有意的现场对齐点，不是待填实现。
