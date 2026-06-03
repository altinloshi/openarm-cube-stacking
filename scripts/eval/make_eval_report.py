"""Create a small Markdown report from an evaluation results.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Create OpenArm cube-stack evaluation report.")
parser.add_argument("results", type=Path, help="Path to logs/eval/openarm_cube_stack/.../results.json")
parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output path.")
args = parser.parse_args()

data = json.loads(args.results.read_text())
lines = [
    "# OpenArm Cube Stack Evaluation Report",
    "",
    f"- Task: `{data.get('task')}`",
    f"- Environments: {data.get('num_envs')}",
    f"- Episodes: {data.get('episodes')}",
    f"- Full stack success rate: {data.get('full_stack_success_rate')}",
    f"- Average cubes stacked: {data.get('average_cubes_stacked')}",
    f"- Per-cube success rate: {data.get('per_cube_success_rate')}",
    f"- Mean final stack error: {data.get('mean_final_stack_error')}",
    f"- Mean grasp retries: {data.get('mean_grasp_retries')}",
    f"- Timeout rate: {data.get('timeout_rate')}",
    "",
]
report = "\n".join(lines)
if args.output is None:
    args.output = args.results.with_suffix('.md')
args.output.write_text(report)
print(report)
print(f"[INFO] Wrote report: {args.output}")
