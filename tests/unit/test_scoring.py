"""分层 outcome reward 评分纯函数测试。"""

import pytest

from siete_rl.models import Verification
from siete_rl.rewards import layered_score, parse_pytest_summary

LAYERED_REWARD_CAP = 0.20


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


@pytest.mark.parametrize("layered_reward_cap", [0.1, 0.2, 0.5])
@pytest.mark.parametrize(
    ("fail_to_pass", "pass_to_pass"),
    [
        ([], []),
        (["a.py::test_fix"], ["b.py::test_old"]),
    ],
)
def test_resolved_is_exactly_one(layered_reward_cap, fail_to_pass, pass_to_pass):
    v = _verification(result="resolved", apply_status="applied", exit_code=0)
    assert layered_score(
        verification=v,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        layered_reward_cap=layered_reward_cap,
    ) == 1.0


def test_apply_failed_is_zero():
    v = _verification(apply_status="apply_failed")
    assert layered_score(
        verification=v, fail_to_pass=["t1"], pass_to_pass=[],
        layered_reward_cap=LAYERED_REWARD_CAP,
    ) == 0.0


def test_check_failed_is_zero():
    v = _verification(apply_status="check_failed")
    assert layered_score(
        verification=v, fail_to_pass=["t1"], pass_to_pass=[],
        layered_reward_cap=LAYERED_REWARD_CAP,
    ) == 0.0


def test_partial_ratio_uses_squared_shape_and_cap():
    # F2P 两个测试过一个：p=0.5 → R = 0.20 * 0.5² = 0.05
    score = layered_score(
        verification=_verification(stdout=PYTEST_STDOUT),
        fail_to_pass=[
            "mypy/test/teststubgen.py::TestStubgenPythonSuite::testGenericClassTypeVarTuple",
            "mypy/test/teststubgen.py::TestStubgenPythonSuite::testGenericClassTypeVarTuple_semanal",
        ],
        pass_to_pass=[],
        layered_reward_cap=LAYERED_REWARD_CAP,
    )
    assert score == pytest.approx(0.05)


def test_any_p2p_regression_cancels_partial_score():
    # p=0.5；P2P 两个过一个 → hard gate 取消部分分
    stdout = (
        PYTEST_STDOUT
        + "PASSED mypy/test/teststubgen.py::TestStubgenPythonSuite::testKept\n"
    )
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=[
            "mypy/test/teststubgen.py::TestStubgenPythonSuite::testGenericClassTypeVarTuple",
            "mypy/test/teststubgen.py::TestStubgenPythonSuite::testGenericClassTypeVarTuple_semanal",
        ],
        pass_to_pass=[
            "mypy/test/teststubgen.py::TestStubgenPythonSuite::testKept",
            "mypy/test/teststubgen.py::TestStubgenPythonSuite::testOtherRegressed",
        ],
        layered_reward_cap=LAYERED_REWARD_CAP,
    )
    assert score == 0.0


def test_all_f2p_passed_but_all_p2p_failed_is_zero():
    # F2P 全过但 P2P 全挂 → 安全门关闭，拿不到部分分
    stdout = """
PASSED a.py::test_target
FAILED a.py::test_regressed - RuntimeError
"""
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["a.py::test_target"],
        pass_to_pass=["a.py::test_regressed"],
        layered_reward_cap=LAYERED_REWARD_CAP,
    )
    assert score == 0.0


def test_regression_in_large_p2p_suite_cancels_partial_score():
    # P2P 是安全门：套件再大，真实回归也取消部分分
    passed_p2p = [f"PASSED tests/test_ec2/test_subnets.py::test_p2p_{i}" for i in range(107)]
    failed_p2p = [f"FAILED tests/test_ec2/test_subnets.py::test_p2p_bad_{i} - RuntimeError" for i in range(6)]
    stdout = (
        "PASSED tests/test_subnets.py::test_f2p_target\n"
        + "\n".join(passed_p2p + failed_p2p)
        + "\n"
    )
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["tests/test_subnets.py::test_f2p_target"],
        pass_to_pass=[f"tests/test_ec2/test_subnets.py::test_p2p_{i}" for i in range(107)]
        + [f"tests/test_ec2/test_subnets.py::test_p2p_bad_{i}" for i in range(6)],
        layered_reward_cap=LAYERED_REWARD_CAP,
    )
    assert score == 0.0


def test_unresolved_without_test_vectors_is_zero():
    score = layered_score(
        verification=_verification(stdout="PASSED a.py::test_x\n"),
        fail_to_pass=[],
        pass_to_pass=[],
        layered_reward_cap=LAYERED_REWARD_CAP,
    )
    assert score == 0.0


def test_unresolved_all_listed_tests_pass_reaches_layered_reward_cap():
    stdout = """
PASSED tests/test_target.py::test_fixed
PASSED tests/test_regression.py::test_preserved
FAILED tests/test_extra.py::test_unexpected - AssertionError
"""
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["tests/test_target.py::test_fixed"],
        pass_to_pass=["tests/test_regression.py::test_preserved"],
        layered_reward_cap=LAYERED_REWARD_CAP,
    )
    assert score == LAYERED_REWARD_CAP


def test_parametrized_and_dotted_names_match_by_last_segment():
    # 一个匹配、一个未通过，p=1/2；类名段不一致仍能匹配。
    stdout = "PASSED tests/test_a.py::RuntimeSuite::test_case[param]\n"
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=[
            "tests/test_a.py::DatasetSuite::test_case[param]",
            "tests/test_a.py::DatasetSuite::test_missing",
        ],
        pass_to_pass=[],
        layered_reward_cap=LAYERED_REWARD_CAP,
    )
    assert score == pytest.approx(0.05)


def test_bare_dataset_entry_falls_back_to_name_only_match():
    # 一个匹配、一个未通过，p=1/2；无文件段条目退化为按裸名匹配。
    stdout = "PASSED tests/test_a.py::TestSuite::test_case[param]\n"
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["test_case[param]", "test_missing"],
        pass_to_pass=[],
        layered_reward_cap=LAYERED_REWARD_CAP,
    )
    assert score == pytest.approx(0.05)


def test_cross_file_same_bare_name_does_not_collide():
    # mypy-14981 反例：同一裸名 testCachedProperty 在 testcheck.py 过（P2P）、
    # 在 teststubgen.py 挂（F2P）。文件段参与匹配后，P2P 的通过不得错记为 F2P 通过。
    stdout = """
PASSED mypy/test/testcheck.py::TypeCheckSuite::check-functools.test::testCachedProperty
FAILED mypy/test/teststubgen.py::StubgenPythonSuite::stubgen.test::testCachedProperty - AssertionError
"""
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=[
            "mypy/test/teststubgen.py::StubgenPythonSuite::stubgen.test::testCachedProperty",
        ],
        pass_to_pass=[
            "mypy/test/testcheck.py::TypeCheckSuite::check-functools.test::testCachedProperty",
        ],
        layered_reward_cap=LAYERED_REWARD_CAP,
    )
    assert score == 0.0  # p=0，P2P 全过也不产生奖励


def test_unseen_p2p_counts_as_not_passed():
    # P2P 测试未出现在 summary（未收集/跳过/整文件 ERROR）按未通过计，方向保守
    stdout = "PASSED a.py::test_target\n"
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["a.py::test_target"],
        pass_to_pass=["a.py::test_not_run"],
        layered_reward_cap=LAYERED_REWARD_CAP,
    )
    assert score == 0.0  # p=1，但 P2P 没有显式通过


def test_whole_file_collection_error_scores_zero():
    # 整文件收集 ERROR：F2P 拿不到分、P2P 也未通过 → R=0（不会出现 k=0 漏扣）
    stdout = "ERROR mypy/test/teststubgen.py - ImportError\n"
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["mypy/test/teststubgen.py::TestStubgenPythonSuite::test_x"],
        pass_to_pass=["mypy/test/teststubgen.py::TestStubgenPythonSuite::test_y"],
        layered_reward_cap=LAYERED_REWARD_CAP,
    )
    assert score == 0.0
