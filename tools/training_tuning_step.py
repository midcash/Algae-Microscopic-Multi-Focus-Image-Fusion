from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autotune.config import DEFAULT_AUTOTUNE_CONFIG
from autotune.engine import run_training_tuning_step
from autotune.file_env import FileLiveEnvironment
from autotune.history import TrainingHistory


def parse_args():
    parser = argparse.ArgumentParser(description="训练调优闭环单步分析入口")
    parser.add_argument("--live-dir", required=True, help="训练实验 live 目录")
    return parser.parse_args()


def main():
    args = parse_args()
    env = FileLiveEnvironment(args.live_dir)
    history = TrainingHistory.load(DEFAULT_AUTOTUNE_CONFIG["history_file"])
    result = run_training_tuning_step(
        env,
        history,
        stale_timeout_sec=float(DEFAULT_AUTOTUNE_CONFIG["stale_timeout_sec"]),
        stale_repeat_threshold=int(DEFAULT_AUTOTUNE_CONFIG["stale_repeat_threshold"]),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
