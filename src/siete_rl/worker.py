from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path

from siete_rl import train
from siete_rl.cli import SignalBoundary, WorkflowTermination
from siete_rl.launcher import RunEndpoints


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="siete_rl.worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--server-host")
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--group-port", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    endpoints = None
    if args.server_host is not None:
        endpoints = RunEndpoints(
            host=args.server_host,
            server_port=args.server_port,
            group_port=args.group_port,
            ddp_port=int(os.environ["MASTER_PORT"]),
        )
    try:
        with SignalBoundary():
            report = train.run_worker(
                args.config,
                run_id=args.run_id,
                vllm_endpoints=endpoints,
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
