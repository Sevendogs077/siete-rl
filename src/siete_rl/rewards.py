from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re
from typing import TYPE_CHECKING, Any

from siete_rl.models import Verification

if TYPE_CHECKING:
    from siete_rl.environment import SWEEnvironment


DEFAULT_LAYERED_REWARD_CAP = 0.20
_SUMMARY_LINE = re.compile(r"(?m)^(PASSED|FAILED|ERROR)\s+(\S+)")
_MatchKey = tuple[str | None, str]


def parse_pytest_summary(stdout: str) -> tuple[set[str], set[str]]:
    passed: set[str] = set()
    failed: set[str] = set()
    for status, nodeid in _SUMMARY_LINE.findall(stdout):
        if status == "PASSED":
            passed.add(nodeid)
        else:
            failed.add(nodeid)
    return passed, failed


def _match_key(test_id: str) -> _MatchKey:
    parts = test_id.split("::")
    file = parts[0].strip() if len(parts) > 1 else None
    return file, parts[-1].strip()


def _keys_match(target: _MatchKey, candidate: _MatchKey) -> bool:
    if target[1] != candidate[1]:
        return False
    return target[0] is None or target[0] == candidate[0]


def _count_matched(targets: set[_MatchKey], pool: set[_MatchKey]) -> int:
    return sum(
        1 for target in targets if any(_keys_match(target, candidate) for candidate in pool)
    )


def layered_score(
    *,
    verification: Verification,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    layered_reward_cap: float,
) -> float:
    if verification.result == "resolved":
        return 1.0
    if verification.patch_apply_status != "applied":
        return 0.0
    passed, _ = parse_pytest_summary(verification.stdout)
    passed_keys = {_match_key(nodeid) for nodeid in passed}
    f2p = {_match_key(test_id) for test_id in fail_to_pass}
    p2p = {_match_key(test_id) for test_id in pass_to_pass}
    if not f2p or _count_matched(p2p, passed_keys) != len(p2p):
        return 0.0
    p = _count_matched(f2p, passed_keys) / len(f2p)
    return layered_reward_cap * p**2


def binary_reward(
    completions: list[object],
    environments: list[SWEEnvironment],
    max_workers: int = 1,
    **kwargs: Any,
) -> list[float | None]:
    del kwargs
    if len(completions) != len(environments):
        raise ValueError("completion and environment counts do not match")

    def finalize(pair: tuple[SWEEnvironment, object]) -> float | None:
        environment, completion = pair
        value = environment._finalize(completion)
        settlement = environment.settlement
        if settlement.status == "infra_error":
            return None
        return value

    if max_workers > 1 and len(environments) > 1:
        with ThreadPoolExecutor(
            max_workers=min(len(environments), max_workers)
        ) as pool:
            return list(
                pool.map(
                    finalize,
                    zip(environments, completions, strict=True),
                )
            )
    return [
        finalize((environment, completion))
        for completion, environment in zip(completions, environments, strict=True)
    ]
