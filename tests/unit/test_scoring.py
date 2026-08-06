"""分层 outcome reward 评分纯函数测试。"""

import math

import pytest

from siete_rl.models import Verification
from siete_rl.scoring import layered_score, parse_pytest_summary

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
