from __future__ import annotations

import re
from typing import Callable

from siete_rl.docker import CommandResult, DockerRuntimeError, DockerSandbox
from siete_rl.models import Evaluation, Verification


class VerificationInfrastructureError(DockerRuntimeError):
    pass


class SWEGymVerifier:
    def __init__(
        self,
        *,
        sandbox_factory: Callable[[], DockerSandbox],
        evaluation: Evaluation,
    ) -> None:
        self.sandbox_factory = sandbox_factory
        self.evaluation = evaluation
        self.cleanup_events: list[dict[str, object]] = []
        self._active_sandbox: DockerSandbox | None = None
        self._pending_verification: Verification | None = None

    def verify(self, patch: str) -> Verification:
        if not patch.strip():
            raise VerificationInfrastructureError("verifier requires a non-empty submitted patch")

        if self._pending_verification is not None:
            self.close()
            verification = self._pending_verification
            self._pending_verification = None
            return verification

        primary: BaseException | None = None
        try:
            sandbox = self.sandbox_factory()
            self._active_sandbox = sandbox
            sandbox.open()
            self._pending_verification = self._verify_in_sandbox(sandbox, patch)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            if self._active_sandbox is not None:
                try:
                    self.close()
                except BaseException as cleanup_exc:
                    if primary is not None:
                        primary.add_note(
                            "verifier container cleanup failed: "
                            f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                        )
                    elif self._pending_verification is None:
                        raise VerificationInfrastructureError(
                            f"verifier container cleanup failed: {cleanup_exc}"
                        ) from cleanup_exc
        if self._pending_verification is None:
            raise VerificationInfrastructureError("verifier produced no result")
        verification = self._pending_verification
        self._pending_verification = None
        return verification

    def close(self) -> None:
        if self._active_sandbox is None:
            return
        sandbox = self._active_sandbox
        try:
            sandbox.close()
        finally:
            self.cleanup_events.append(
                {
                    "scope": "verifier",
                    "task_id": sandbox.environment.task_id,
                    "container_name": sandbox.container_name,
                    "container_id": sandbox.acquired_container_id,
                    "operations": sandbox.drain_cleanup_operations(),
                    "residual": sandbox.container_id is not None,
                }
            )
            if sandbox.container_id is None:
                self._active_sandbox = None

    def drain_cleanup_events(self) -> list[dict[str, object]]:
        events = list(self.cleanup_events)
        self.cleanup_events.clear()
        return events

    def _verify_in_sandbox(self, sandbox: DockerSandbox, patch: str) -> Verification:
        checked = sandbox.exec(
            [
                "git",
                "-C",
                sandbox.environment.workdir,
                "apply",
                "--check",
                "--whitespace=nowarn",
                "-",
            ],
            input_text=patch,
        )
        if checked.timed_out:
            raise VerificationInfrastructureError("git apply --check timed out")
        if checked.exit_code != 0:
            return _verification("unresolved", "check_failed", False, checked)

        applied = sandbox.exec(
            [
                "git",
                "-C",
                sandbox.environment.workdir,
                "apply",
                "--whitespace=nowarn",
                "-",
            ],
            input_text=patch,
        )
        if applied.timed_out:
            raise VerificationInfrastructureError("git apply timed out")
        if applied.exit_code != 0:
            return _verification("unresolved", "apply_failed", False, applied)

        evaluated = sandbox.exec(
            ["/bin/bash", "-s"],
            input_text=self.evaluation.offline_eval_script,
            timeout_sec=sandbox.environment.verifier_timeout_sec,
        )
        pytest_started = _pytest_started(evaluated.stdout, evaluated.stderr)
        evidence = "\n".join(
            part
            for part in (
                evaluated.stdout.strip()[-2000:],
                evaluated.stderr.strip()[-2000:],
            )
            if part
        )
        detail = f": {evidence}" if evidence else ""
        if evaluated.timed_out:
            if not pytest_started:
                raise VerificationInfrastructureError(
                    "offline evaluation timed out before pytest started" + detail
                )
            return _verification("unresolved", "applied", True, evaluated)
        if not pytest_started:
            raise VerificationInfrastructureError(
                "offline evaluator finished without the required pytest start marker"
                + detail
            )
        result = "resolved" if evaluated.exit_code == 0 else "unresolved"
        return _verification(result, "applied", True, evaluated)


def _verification(
    result: str,
    patch_apply_status: str,
    pytest_started: bool,
    evidence: CommandResult,
) -> Verification:
    return Verification.model_validate(
        {
            "result": result,
            "patch_apply_status": patch_apply_status,
            "pytest_started": pytest_started,
            "exit_code": evidence.exit_code,
            "stdout": evidence.stdout,
            "stderr": evidence.stderr,
        }
    )


def _pytest_started(stdout: str, stderr: str) -> bool:
    return re.search(r"(?m)^\+\s+pytest(?:\s|$)", stdout + "\n" + stderr) is not None
