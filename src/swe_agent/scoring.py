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

# 分层 reward 默认超参；config.py 与 environment.py 共用，避免两处默认值漂移
DEFAULT_LAMBDA = 8.0
DEFAULT_MU = math.log(2)  # ln2：每个 P2P 回归减半


def parse_pytest_summary(stdout: str) -> tuple[set[str], set[str]]:
    """解析 `pytest -rA` short summary，返回 (passed nodeids, failed nodeids)。

    FAILED 行可能带 " - reason" 后缀，`\\S+` 在空格处截断，不会带入。
    ERROR 计入 failed（收集/fixture 错误等价于测试失败）。
    整文件收集 ERROR（`ERROR path/to/test_file.py`，无 `::`）归一后是路径，
    与任何裸测试名都不匹配：F2P 测试因此拿不到分（p 偏低，保守方向），
    但 P2P 模块 import 挂掉时 k=0，不扣回归分。
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
    """按分层公式给 applied-but-unresolved 的 patch 打部分分。

    裸名集合匹配的碰撞语义（有意接受的权衡）：
      a) F2P 清单里归一后同名的条目会被集合去重，分母变小，p 被抬高；
      b) 同一裸名在 A 文件过、在 B 文件挂时会同时进入 passed_names 与
         failed_names，p 与 k 各自计一次；
      c) 多个挂掉的 P2P 测试若共享裸名，只算 k=1。
    之所以接受：数据集给的 nodeid 与运行时 nodeid 本来就不能文本匹配
    （如 mypy 数据集写 `StubgenPythonSuite::stubgen.test::testX`，运行时
    是 `TestStubgenPythonSuite::testX`），只能按最后一段对齐。
    """

    if verification.result == "resolved":
        return 1.0
    if verification.patch_apply_status != "applied" or not verification.pytest_started:
        # pytest_started 检查是防御性的：Verification.validate_evidence 已保证
        # applied ⇒ pytest_started，这里防未来模型约束变动。
        return 0.0
    passed, failed = parse_pytest_summary(verification.stdout)
    passed_names = {_normalize_test_name(n) for n in passed}
    failed_names = {_normalize_test_name(n) for n in failed}
    f2p = {_normalize_test_name(t) for t in fail_to_pass}
    p2p = {_normalize_test_name(t) for t in pass_to_pass}
    p = len(f2p & passed_names) / len(f2p) if f2p else 1.0
    k = len(p2p & failed_names)
    return math.expm1(lambda_ * p) / math.expm1(lambda_) * math.exp(-mu * k)
