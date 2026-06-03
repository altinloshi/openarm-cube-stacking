"""Generate a human-readable evaluation report from results.json.

Usage
-----
    python scripts/eval/make_eval_report.py \\
        logs/eval/openarm_cube_stack/2026-06-01_12-00-00/results.json

    # Or pass a directory to find the latest results.json
    python scripts/eval/make_eval_report.py \\
        logs/eval/openarm_cube_stack/
"""

import argparse
import json
import os
import sys
from pathlib import Path


def find_latest_results(base_dir: str) -> str | None:
    """Find the most recently modified results.json under base_dir."""
    results = sorted(
        Path(base_dir).rglob("results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(results[0]) if results else None


def print_report(metrics: dict) -> None:
    sep = "─" * 58
    print(sep)
    print("  OpenArm Cube Stacking – Evaluation Report")
    print(sep)
    print(f"  Task              : {metrics.get('task', 'N/A')}")
    print(f"  Environments      : {metrics.get('num_envs', 'N/A')}")
    print(f"  Episodes          : {metrics.get('num_episodes', 'N/A')}")
    print(sep)
    print(f"  Full stack success: {metrics.get('full_stack_success_rate', 0.0):.1%}")
    print(f"  Avg cubes stacked : {metrics.get('average_cubes_stacked', 0.0):.2f} / 5")
    print()
    pcs = metrics.get("per_cube_success_rate", [0.0] * 5)
    for i, rate in enumerate(pcs):
        bar = "█" * int(rate * 20)
        print(f"  Cube {i + 1} success   : {rate:.1%}  {bar}")
    print()
    print(f"  Cube drop rate    : {metrics.get('cube_drop_rate', 0.0):.1%}")
    print(f"  Stack collapse    : {metrics.get('stack_collapse_rate', 0.0):.1%}")
    print(f"  Mean stack error  : {metrics.get('mean_final_stack_error', 0.0):.4f} m")
    print(f"  Mean ep length    : {metrics.get('mean_episode_length_s', 0.0):.1f} s")
    print(f"  Mean retries      : {metrics.get('mean_grasp_retries', 0.0):.2f}")
    print(f"  Timeout rate      : {metrics.get('timeout_rate', 0.0):.1%}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print evaluation report from results.json.")
    parser.add_argument("path", help="Path to results.json or a directory containing it.")
    args = parser.parse_args()

    path = args.path
    if os.path.isdir(path):
        path = find_latest_results(path)
        if path is None:
            print(f"[ERROR]: No results.json found under {args.path}")
            sys.exit(1)

    with open(path) as f:
        metrics = json.load(f)

    print_report(metrics)


if __name__ == "__main__":
    main()
