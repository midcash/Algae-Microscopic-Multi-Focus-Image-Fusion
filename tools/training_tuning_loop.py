from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autotune.engine import run_training_tuning_loop
from autotune.file_env import FileLiveEnvironment


def parse_args():
    parser = argparse.ArgumentParser(description="训练调优闭环多轮监听入口")
    parser.add_argument("--live-dir", required=True, help="训练实验 live 目录")
    parser.add_argument("--max-rounds", type=int, default=10, help="最大轮询轮数")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="轮询间隔（秒）")
    return parser.parse_args()


def main():
    args = parse_args()
    env = FileLiveEnvironment(args.live_dir)
    result = run_training_tuning_loop(
        env,
        config={
            "max_rounds": args.max_rounds,
            "poll_interval_sec": args.poll_interval,
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
