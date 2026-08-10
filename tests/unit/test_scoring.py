"""分层 outcome reward 评分纯函数测试。"""

import math

import pytest

from siete_rl.models import Verification
from siete_rl.scoring import layered_score, parse_pytest_summary

LAMBDA = 8.0


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
        lambda_=LAMBDA,
    ) == 1.0


def test_apply_failed_is_zero():
    v = _verification(apply_status="apply_failed")
    assert layered_score(
        verification=v, fail_to_pass=["t1"], pass_to_pass=[],
        lambda_=LAMBDA,
    ) == 0.0


def test_check_failed_is_zero():
    v = _verification(apply_status="check_failed")
    assert layered_score(
        verification=v, fail_to_pass=["t1"], pass_to_pass=[],
        lambda_=LAMBDA,
    ) == 0.0


def test_partial_ratio_uses_exponential_shape():
    # F2P 两个测试过一个：p=0.5，λ=8 → expm1(4)/expm1(8) ≈ 0.018
    score = layered_score(
        verification=_verification(stdout=PYTEST_STDOUT),
        fail_to_pass=[
            "mypy/test/teststubgen.py::TestStubgenPythonSuite::testGenericClassTypeVarTuple",
            "mypy/test/teststubgen.py::TestStubgenPythonSuite::testGenericClassTypeVarTuple_semanal",
        ],
        pass_to_pass=[],
        lambda_=LAMBDA,
    )
    assert score == pytest.approx(math.expm1(4.0) / math.expm1(8.0), rel=1e-9)
    assert score < 0.02  # 线性会给 0.5，指数型必须压到接近 0


def test_p2p_keep_rate_discounts_partial_score():
    # p=0.5；P2P 两个过一个（q=0.5）→ R = f(0.5) · 0.5
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
        lambda_=LAMBDA,
    )
    base = math.expm1(4.0) / math.expm1(8.0)
    assert score == pytest.approx(base * 0.5, rel=1e-9)


def test_all_f2p_passed_but_all_p2p_failed_is_zero():
    # F2P 全过但 P2P 全挂 → q=0，「只修新测试不管回归」拿不到部分分
    stdout = """
PASSED a.py::test_target
FAILED a.py::test_regressed - RuntimeError
"""
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["a.py::test_target"],
        pass_to_pass=["a.py::test_regressed"],
        lambda_=LAMBDA,
    )
    assert score == 0.0


def test_single_regression_in_large_p2p_suite_costs_little():
    # 回归率语义：113 个 P2P 挂 6 个 → q=107/113，不再按个数指数归零
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
        lambda_=LAMBDA,
    )
    assert score == pytest.approx(107 / 113, rel=1e-9)


def test_unresolved_without_test_vectors_is_zero():
    score = layered_score(
        verification=_verification(stdout="PASSED a.py::test_x\n"),
        fail_to_pass=[],
        pass_to_pass=[],
        lambda_=LAMBDA,
    )
    assert score == 0.0


def test_unresolved_all_listed_tests_pass_but_verifier_fails_is_zero():
    stdout = """
PASSED tests/test_target.py::test_fixed
PASSED tests/test_regression.py::test_preserved
FAILED tests/test_extra.py::test_unexpected - AssertionError
"""
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["tests/test_target.py::test_fixed"],
        pass_to_pass=["tests/test_regression.py::test_preserved"],
        lambda_=LAMBDA,
    )
    assert score == 0.0


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
        lambda_=LAMBDA,
    )
    assert score == pytest.approx(math.expm1(4.0) / math.expm1(8.0), rel=1e-9)


def test_bare_dataset_entry_falls_back_to_name_only_match():
    # 一个匹配、一个未通过，p=1/2；无文件段条目退化为按裸名匹配。
    stdout = "PASSED tests/test_a.py::TestSuite::test_case[param]\n"
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["test_case[param]", "test_missing"],
        pass_to_pass=[],
        lambda_=LAMBDA,
    )
    assert score == pytest.approx(math.expm1(4.0) / math.expm1(8.0), rel=1e-9)


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
        lambda_=LAMBDA,
    )
    assert score == 0.0  # p=0、q=1 → R=0；旧裸名匹配会白捡 f(1/1)·2^-1


def test_unseen_p2p_counts_as_not_passed():
    # P2P 测试未出现在 summary（未收集/跳过/整文件 ERROR）按未通过计，方向保守
    stdout = "PASSED a.py::test_target\n"
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["a.py::test_target"],
        pass_to_pass=["a.py::test_not_run"],
        lambda_=LAMBDA,
    )
    assert score == 0.0  # p=1 但 q=0/1


def test_whole_file_collection_error_scores_zero():
    # 整文件收集 ERROR：F2P 拿不到分、P2P 也未通过 → R=0（不会出现 k=0 漏扣）
    stdout = "ERROR mypy/test/teststubgen.py - ImportError\n"
    score = layered_score(
        verification=_verification(stdout=stdout),
        fail_to_pass=["mypy/test/teststubgen.py::TestStubgenPythonSuite::test_x"],
        pass_to_pass=["mypy/test/teststubgen.py::TestStubgenPythonSuite::test_y"],
        lambda_=LAMBDA,
    )
    assert score == 0.0
