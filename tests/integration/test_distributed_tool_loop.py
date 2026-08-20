from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from trl import GRPOTrainer

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


class FakeColocatedVLLM:
    enable_sleep_mode = True

    def sync_weights(self) -> None:
        pass

    def generate(self, *, prompts, images, num_generations, profiler=None):
        del num_generations, profiler
        gathered_prompts = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_prompts, prompts)
        self.gathered_sizes = [len(batch) for batch in gathered_prompts]
        completion_ids = [[prompt[0] + 100] for prompt in prompts]
        return prompts, completion_ids, [[[float(ids[0])]] for ids in completion_ids], None


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


def colocate_worker(result_dir: Path) -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    counts = [2, 1, 0, 1]
    prompts = [[rank * 10 + index + 1] for index in range(counts[rank])]
    images = [[f"image-{rank}-{index}"] for index in range(counts[rank])] or None
    trainer = make_trainer(rank)
    trainer.vllm_mode = "colocate"
    trainer._tokenizer = SimpleNamespace(eos_token_id=99)
    trainer.vllm_generation = FakeColocatedVLLM()

    completion_ids, _ = trainer._generate_tool_loop_turn(prompts, images, {})

    (result_dir / f"colocate-rank-{rank}.json").write_text(
        json.dumps(
            {
                "completion_ids": completion_ids,
                "gathered_sizes": trainer.vllm_generation.gathered_sizes,
            }
        )
    )
    dist.destroy_process_group()


def loss_worker(result_dir: Path) -> None:
    dist.init_process_group("gloo", timeout=timedelta(seconds=5))
    rank = dist.get_rank()
    trainer = object.__new__(SWEGRPOTrainer)
    trainer.model = SimpleNamespace(training=True)
    trainer._metrics = {"train": {"clip_ratio": []}, "eval": {}}
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))

    def parent_loss(self, unwrapped_model, inputs):
        marker = torch.tensor(float(rank))
        dist.all_reduce(marker)
        self._metrics["train"]["clip_ratio"].append(marker.item())
        return sum(
            (parameter.reshape(-1)[0] * 0.0 for parameter in unwrapped_model.parameters()),
            inputs["token_weights"].new_zeros(()),
        )

    GRPOTrainer.compute_liger_loss = parent_loss
    token_weights = torch.zeros(1, 2) if rank == 1 else torch.ones(1, 2)
    loss = trainer.compute_liger_loss(
        model,
        {"completion_mask": torch.ones(1, 2), "token_weights": token_weights},
    )
    loss.backward()
    (result_dir / f"loss-rank-{rank}.json").write_text(
        json.dumps(
            {
                "clip_ratio": trainer._metrics["train"]["clip_ratio"],
                "zero_gradients": all(
                    parameter.grad is not None
                    and torch.count_nonzero(parameter.grad).item() == 0
                    for parameter in model.parameters()
                ),
            }
        )
    )
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


def test_colocate_tp4_preserves_uneven_active_prompts(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        __file__,
        "--colocate-worker",
        str(tmp_path),
    ]
    subprocess.run(command, check=True, timeout=30)

    results = [
        json.loads((tmp_path / f"colocate-rank-{rank}.json").read_text())
        for rank in range(4)
    ]
    assert [result["completion_ids"] for result in results] == [
        [[101], [102]],
        [[111]],
        [],
        [[131]],
    ]
    assert [result["gathered_sizes"] for result in results] == [[2, 2, 2, 2]] * 4


def test_fully_censored_rank_keeps_loss_collective_order(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        __file__,
        "--loss-worker",
        str(tmp_path),
    ]
    subprocess.run(command, check=True, timeout=30)

    results = [
        json.loads((tmp_path / f"loss-rank-{rank}.json").read_text())
        for rank in range(2)
    ]
    assert [result["clip_ratio"] for result in results] == [[1.0], [1.0]]
    assert all(result["zero_gradients"] for result in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--colocate-worker", type=Path)
    parser.add_argument("--loss-worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.colocate_worker:
        colocate_worker(args.colocate_worker)
    else:
        loss_worker(args.loss_worker)
