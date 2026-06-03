"""Export a trained RSL-RL policy to best_policy/ directory.

Copies the latest (or specified) checkpoint to:
  best_policy/best_policy.pt

Exports TorchScript and ONNX formats to:
  best_policy/exported/policy.pt
  best_policy/exported/policy.onnx

Usage
-----
    python scripts/rsl_rl/export_policy.py \\
        --task=Nepher-OpenArm-CubeStack-LL-Play-v0

    # Specify a checkpoint explicitly
    python scripts/rsl_rl/export_policy.py \\
        --task=Nepher-OpenArm-CubeStack-LL-Play-v0 \\
        --checkpoint=logs/rsl_rl/openarm_cube_stack_ll/2026-06-01_12-00-00/model_3000.pt
"""

import argparse
import os
import shutil
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Export an RSL-RL policy for OpenArm cube stacking.")
parser.add_argument("--task", type=str, default="Nepher-OpenArm-CubeStack-LL-Play-v0", help="Gym task ID.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (1 for export).")
parser.add_argument("--output_dir", type=str, default="best_policy", help="Output directory.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import openarm_cube_stacking.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO]: Loading checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # ── copy checkpoint ───────────────────────────────────────────────────────
    out_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    best_pt = os.path.join(out_dir, "best_policy.pt")
    shutil.copy2(resume_path, best_pt)
    print(f"[INFO]: Copied checkpoint to: {best_pt}")

    # ── export TorchScript ────────────────────────────────────────────────────
    export_dir = os.path.join(out_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()

    # TorchScript
    ts_path = os.path.join(export_dir, "policy.pt")
    try:
        scripted = torch.jit.script(policy)
        scripted.save(ts_path)
        print(f"[INFO]: TorchScript exported to: {ts_path}")
    except Exception as e:
        print(f"[WARN]: TorchScript export failed ({e}); trying torch.jit.trace")
        try:
            traced = torch.jit.trace(policy, obs)
            traced.save(ts_path)
            print(f"[INFO]: TorchScript (traced) exported to: {ts_path}")
        except Exception as e2:
            print(f"[WARN]: TorchScript trace also failed: {e2}")

    # ONNX
    onnx_path = os.path.join(export_dir, "policy.onnx")
    try:
        torch.onnx.export(
            policy,
            obs,
            onnx_path,
            input_names=["obs"],
            output_names=["actions"],
            opset_version=17,
        )
        print(f"[INFO]: ONNX exported to: {onnx_path}")
    except Exception as e:
        print(f"[WARN]: ONNX export failed: {e}")

    env.close()
    print("[INFO]: Export complete.")


if __name__ == "__main__":
    main()
    simulation_app.close()
