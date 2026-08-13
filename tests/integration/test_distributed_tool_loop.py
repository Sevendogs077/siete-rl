from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch.distributed as dist

import siete_rl.trainer as trainer_module
from siete_rl.process_mask import TurnRecord
from siete_rl.train import _gather_rollout_records
from siete_rl.trainer import SWEGRPOTrainer, _global_active_counts


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompts, **kwargs):
        del kwargs
        self.calls += 1
        offset = self.calls * 100
        return {
            "prompt_ids": prompts,
            "completion_ids": [[prompt[0] + offset, 99] for prompt in prompts],
            "logprobs": [
                [[float(prompt[0] + offset)], [0.0]] for prompt in prompts
            ],
            "logprob_token_ids": None,
        }


class Environment:
    def __init__(self) -> None:
        self.terminated = False
        self.loop_exit = None
        self.turn_records: list[TurnRecord] = []
        self._steps = []

    def record_step(self, *, terminate: bool = False) -> str:
        self._steps.append(object())
        self.terminated = terminate
        return "done"

    def _record_loop_exit(self, reason) -> None:
        self.loop_exit = reason


def tool_call(name: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": name, "arguments": {}}}
        ],
    }


def make_trainer(rank: int) -> SWEGRPOTrainer:
    trainer = object.__new__(SWEGRPOTrainer)
    trainer.use_vllm = True
    trainer.vllm_mode = "server"
    trainer.state = SimpleNamespace(global_step=0)
    trainer._last_loaded_step = 0
    trainer.args = SimpleNamespace(report_to=[])
    trainer.accelerator = SimpleNamespace(is_main_process=rank == 0, process_index=rank)
    trainer.vllm_generation = SimpleNamespace(
        vllm_client=FakeClient(),
        temperature=1.0,
        top_p=1.0,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.1,
        max_completion_length=8,
        logprobs=1,
        structured_outputs_regex=None,
        generation_kwargs={},
    )
    return trainer


def worker(result_dir: Path) -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    gathered_records = _gather_rollout_records(
        [{"rank": rank, "local_slot": 0}]
    )
    trainer = make_trainer(rank)

    rounds = []
    for prompts in ([[rank + 1]], [] if rank == 0 else [[22]]):
        counts = _global_active_counts(len(prompts))
        completion_ids, _ = trainer._generate_tool_loop_turn(prompts, None, {})
        rounds.append({"counts": counts, "completion_ids": completion_ids})

    loop_trainer = make_trainer(rank)
    environment = Environment()
    loop_trainer.environments = [environment]
    loop_trainer.max_tool_calling_iterations = 4
    loop_trainer.max_consecutive_protocol_errors = 2
    loop_trainer.max_completion_length = 32
    loop_trainer._tool_parallel_workers = 1
    loop_trainer._is_vlm = False
    loop_trainer.model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=512)
    )
    loop_trainer._tokenizer = SimpleNamespace(eos_token_id=99, pad_token_id=0)
    loop_trainer._get_tool_suffix_ids = lambda messages: [90] * len(messages)
    loop_trainer._sync_tool_dicts = [
        {
            "bash": environment.record_step,
            "finish": lambda: environment.record_step(terminate=True),
        }
    ]
    loop_trainer._async_tool_dicts = [{}]

    def parse_response(_tokenizer, ids, **kwargs):
        del kwargs
        return tool_call("finish" if ids[0] in (101, 202) else "bash")

    trainer_module.parse_response = parse_response
    _, _, loop_ids, _, _, _, _ = loop_trainer._tool_call_loop(
        prompts=[[{"role": "user", "content": "fix"}]],
        prompt_ids=[[rank + 1]],
        completion_ids=[[11 + rank, 99]],
        completions=[[tool_call("bash")]],
        logprobs=[[0.0, 0.0]],
        images=None,
        multimodal_fields={},
    )
    payload = {
        "rounds": rounds,
        "loop_ids": loop_ids,
        "steps": len(environment._steps),
        "gathered_records": gathered_records,
    }
    (result_dir / f"rank-{rank}.json").write_text(json.dumps(payload))
    dist.destroy_process_group()


def test_uneven_post_tool_generation_preserves_rank_lineage(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        __file__,
        "--worker",
        str(tmp_path),
    ]
    subprocess.run(command, check=True, timeout=30)
    rank0 = json.loads((tmp_path / "rank-0.json").read_text())
    rank1 = json.loads((tmp_path / "rank-1.json").read_text())

    assert rank0["rounds"][1] == {"counts": [0, 1], "completion_ids": []}
    assert rank1["rounds"][1] == {"counts": [0, 1], "completion_ids": [[222, 99]]}
    assert rank0["rounds"][0]["completion_ids"] == [[101, 99]]
    assert rank1["rounds"][0]["completion_ids"] == [[102, 99]]
    assert 202 not in rank0["loop_ids"][0]
    assert 202 in rank1["loop_ids"][0]
    assert rank0["steps"] == 2
    assert rank1["steps"] == 3
    assert rank0["gathered_records"] == [
        {"rank": 0, "local_slot": 0, "global_slot": 0},
        {"rank": 1, "local_slot": 0, "global_slot": 1},
    ]
    assert rank1["gathered_records"] == []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    worker(args.worker)
