"""由 supervisor 启动的、唯一允许初始化 CUDA 的训练 worker。"""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

from swe_agent import train
from swe_agent.cli import SignalBoundary, WorkflowTermination
from swe_agent.launcher import VLLMEndpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swe_agent.worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--server-host", required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--group-port", type=int, required=True)
    parser.add_argument("--trainer-gpu", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    endpoints = VLLMEndpoints(
        host=args.server_host,
        server_port=args.server_port,
        group_port=args.group_port,
    )
    try:
        with SignalBoundary():
            report = train.run_worker(
                args.config,
                run_id=args.run_id,
                vllm_endpoints=endpoints,
                trainer_gpu=args.trainer_gpu,
            )
    except train.TrainingInterrupted as exc:
        return 128 + exc.signum
    except WorkflowTermination as exc:
        return 128 + exc.signum
    except KeyboardInterrupt:
        return 128 + signal.SIGINT
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
