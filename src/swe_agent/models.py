"""第一阶段 SWE 闭环唯一的严格领域模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Termination = Literal[
    "submitted",
    "iteration_cap",
    "context_overlong",
    "format_exhausted",
    "infra_error",
]

LoopExit = Literal["iteration_cap", "context_overlong", "format_exhausted"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Task(StrictModel):
    task_id: str = Field(min_length=1)
    repo_name: str = Field(min_length=1)
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    problem_statement: str = Field(min_length=1)


class Environment(StrictModel):
    environment_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    image_name: str = Field(min_length=1)
    expected_image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    expected_registry_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    workdir: str = Field(min_length=1)
    cpus: float = Field(gt=0)
    memory: str = Field(min_length=2)
    pids_limit: int = Field(gt=0)
    exec_timeout_sec: int = Field(gt=0)
    verifier_timeout_sec: int = Field(gt=0)


class Evaluation(StrictModel):
    """只在进程内交给 verifier 的私有运行时事实。"""

    offline_eval_script: str = Field(min_length=1)


class Sample(StrictModel):
    task: Task
    environment: Environment

    @model_validator(mode="after")
    def validate_links(self) -> "Sample":
        if self.environment.task_id != self.task.task_id:
            raise ValueError("Environment.task_id does not match Task.task_id")
        return self


class Action(StrictModel):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]


class Observation(StrictModel):
    text: str
    exit_code: int | None = None
    error_type: Literal["tool_error"] | None = None
    timed_out: bool = False
    truncated: bool = False


class Step(StrictModel):
    index: int = Field(ge=0)
    action: Action
    observation: Observation


class TerminalEvent(StrictModel):
    """回合终止事件：由终止动作本身在发生时刻记录，而非事后从过程状态推导。"""

    kind: Literal["submitted", "infra_error"]
    step_index: int = Field(ge=0)


class Trajectory(StrictModel):
    task_id: str = Field(min_length=1)
    environment_id: str = Field(min_length=1)
    steps: list[Step]
    termination: Termination

    @model_validator(mode="after")
    def validate_step_order(self) -> "Trajectory":
        indexes = [step.index for step in self.steps]
        if indexes != list(range(len(indexes))):
            raise ValueError("trajectory Step.index values must be contiguous from zero")
        return self


class Verification(StrictModel):
    result: Literal["resolved", "unresolved"]
    patch_apply_status: Literal["check_failed", "apply_failed", "applied"]
    pytest_started: bool
    exit_code: int
    stdout: str
    stderr: str

    @model_validator(mode="after")
    def validate_evidence(self) -> "Verification":
        if self.result == "resolved":
            if self.patch_apply_status != "applied" or not self.pytest_started or self.exit_code != 0:
                raise ValueError("resolved verification requires an applied patch and successful pytest evidence")
        elif self.patch_apply_status == "applied" and not self.pytest_started:
            raise ValueError("an applied unresolved verification requires real pytest evidence")
        return self
