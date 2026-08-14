"""siete_rl 唯一正式命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from siete_rl import prepare, train
from siete_rl.config import load_config


class WorkflowTermination(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"收到 {signal.Signals(signum).name}")
        self.signum = signum


class SignalBoundary:
    def __init__(self) -> None:
        self._triggered = False
        self._previous: dict[signal.Signals, object] = {}

    def __enter__(self) -> "SignalBoundary":
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)

    def _handle(self, signum: int, frame: object) -> None:
        del frame
        if self._triggered:
            print(
                "\n第二次中断：强制退出（不保证清理完成）。"
                "孤儿容器请用 docker ps -aq --filter label=swe_agent.run_id=<run_id> 检查并 docker rm -f",
                file=sys.stderr,
                flush=True,
            )
            os._exit(128 + signum)
        self._triggered = True
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise WorkflowTermination(signum)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="siete-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    grpo = subparsers.add_parser("grpo", help="运行固定 SWE-Gym GRPO 作业")
    grpo.add_argument("--config", type=Path, required=True)
    prepare_parser = subparsers.add_parser("prepare", help="准备 Stage 1/2 训练课程")
    prepare_parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        config, project_root, _ = load_config(args.config)
        output = prepare.prepare_training(project_root, config.runtime.base_seed)
        print(output)
        return 0
    try:
        with SignalBoundary():
            report = train.run(args.config)
    except train.TrainingInterrupted as exc:
        print(json.dumps(exc.report, ensure_ascii=False, indent=2))
        return 128 + exc.signum
    except train.RuntimeNotQualifiedError as exc:
        print(f"runtime qualification rejected: {exc}", file=sys.stderr)
        return 2
    except (ValueError, train.TrainingNotReadyError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except WorkflowTermination as exc:
        return 128 + exc.signum
    except KeyboardInterrupt:
        return 128 + signal.SIGINT
    print(json.dumps(report, ensure_ascii=False, indent=2))
    status = report.get("status")
    if status == "failed":
        return 1
    if status == "interrupted":
        signum = report.get("interrupted_signum", signal.SIGINT)
        return 128 + int(signum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
