"""分层 outcome reward：指数型 F2P 完成度 × P2P 保留率纯函数。

设计来源：《Signal Reshaping for GRPO in Weak-Feedback Agentic Code Repair》
贡献点 1（分层结果奖励）的 test-oracle 改造版——本项目 verifier 直接运行
pytest，用真实测试结果替代论文中的 LLM judge 语义判定。

公式：R = f(p) · q，f(p) = (e^(λp) − 1)/(e^λ − 1)
  p = FAIL_TO_PASS 通过比例（新功能完成程度）；
  q = PASS_TO_PASS 通过比例（旧功能保留程度），未出现在 summary 里的
  P2P 测试（未收集/跳过/整文件 ERROR）按未通过计，方向保守。
只有 verifier resolved 时 R = 1；unresolved 即使清单内 p=q=1，也按 verifier
最终结论返回 0。F2P 全过但 P2P 全挂时 q=0，「只修新测试不管回归」
拿不到部分分；大套件里个别回归按比率折扣，不会像 hard gate 那样直接归零。
只重塑 reward 输入信号；GRPO 目标函数与 advantage 计算不变（论文贡献点 4）。
"""

from __future__ import annotations

import math
import re

from siete_rl.models import Verification

_SUMMARY_LINE = re.compile(r"(?m)^(PASSED|FAILED|ERROR)\s+(\S+)")

# 分层 reward 默认超参；config.py 与 environment.py 共用，避免两处默认值漂移
DEFAULT_LAMBDA = 8.0

# 匹配键：(文件路径, 裸测试名)。文件段为 None 表示清单条目未给出文件，
# 仅按裸名匹配（兼容无路径的旧格式清单）。
_MatchKey = tuple[str | None, str]


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


def _match_key(test_id: str) -> _MatchKey:
    """归一匹配键：(文件路径, 裸测试名)。

    文件段取首个 "::" 之前——数据集条目与 pytest 运行时 nodeid 都带文件
    路径；裸名取最后一段并保留参数化后缀。中间段（suite/类名）在数据集
    与运行时之间可能对不齐（如 mypy 数据集写 `StubgenPythonSuite`、运行
    时是 `TestStubgenPythonSuite`），故不参与匹配。
    """

    parts = test_id.split("::")
    file = parts[0].strip() if len(parts) > 1 else None
    return file, parts[-1].strip()


def _keys_match(target: _MatchKey, candidate: _MatchKey) -> bool:
    """target 来自 F2P/P2P 清单，candidate 来自运行时 summary。

    文件段必须精确相等；target 无文件段时退化为仅按裸名匹配。
    """

    if target[1] != candidate[1]:
        return False
    return target[0] is None or target[0] == candidate[0]


def _count_matched(targets: set[_MatchKey], pool: set[_MatchKey]) -> int:
    return sum(
        1 for target in targets if any(_keys_match(target, c) for c in pool)
    )


def layered_score(
    *,
    verification: Verification,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    lambda_: float,
) -> float:
    """resolved 固定返回 1；仅对 applied-but-unresolved patch 计算部分分。

    匹配按 (文件路径, 裸测试名) 元组，跨文件同名测试不再互相认领
    （旧裸名匹配曾把 A 文件的 P2P 通过错记为 B 文件的 F2P 通过，并把
    F2P 失败错记为 P2P 回归，见 run 20260807T034912Z-91b6 的
    python__mypy-14981 反例）。清单条目无文件段时退化为仅按裸名匹配。
    """

    if verification.result == "resolved":
        return 1.0
    if verification.patch_apply_status != "applied" or not verification.pytest_started:
        # pytest_started 检查是防御性的：Verification.validate_evidence 已保证
        # applied ⇒ pytest_started，这里防未来模型约束变动。
        return 0.0
    passed, failed = parse_pytest_summary(verification.stdout)
    del failed  # 回归按 P2P 保留率 q 计量，不直接消费 failed 集合
    passed_keys = {_match_key(n) for n in passed}
    f2p = {_match_key(t) for t in fail_to_pass}
    p2p = {_match_key(t) for t in pass_to_pass}
    p = _count_matched(f2p, passed_keys) / len(f2p) if f2p else 1.0
    q = _count_matched(p2p, passed_keys) / len(p2p) if p2p else 1.0
    if p == 1.0 and q == 1.0:
        return 0.0
    return math.expm1(lambda_ * p) / math.expm1(lambda_) * q
