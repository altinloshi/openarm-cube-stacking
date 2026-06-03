from __future__ import annotations

import os
import shutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
BEST_POLICY_DIR = os.path.join(PROJECT_ROOT, "best_policy")
BEST_POLICY_CHECKPOINT = os.path.join(BEST_POLICY_DIR, "best_policy.pt")
BEST_POLICY_EXPORT_DIR = os.path.join(BEST_POLICY_DIR, "exported")


def log_root_path(experiment_name: str) -> str:
    return os.path.join(PROJECT_ROOT, "logs", "rsl_rl", experiment_name)


def _find_latest_in_logs(experiment_name: str, load_run: str | None, load_checkpoint: str | None) -> str | None:
    from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

    root = log_root_path(experiment_name)
    if not os.path.isdir(root):
        return None
    try:
        return get_checkpoint_path(root, load_run, load_checkpoint)
    except ValueError:
        return None


def sync_best_policy(
    experiment_name: str,
    load_run: str | None,
    load_checkpoint: str | None,
    *,
    explicit_checkpoint: str | None = None,
) -> str:
    """Copy the selected LL checkpoint into ``best_policy/best_policy.pt``."""
    os.makedirs(BEST_POLICY_DIR, exist_ok=True)
    source = None
    if explicit_checkpoint:
        if os.path.isfile(explicit_checkpoint):
            source = os.path.abspath(explicit_checkpoint)
        else:
            from isaaclab.utils.assets import retrieve_file_path

            source = retrieve_file_path(explicit_checkpoint)
    else:
        source = _find_latest_in_logs(experiment_name, load_run, load_checkpoint)

    if source is not None:
        source = os.path.abspath(source)
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Checkpoint file not found: {source}")
        shutil.copy2(source, BEST_POLICY_CHECKPOINT)
        print(f"[INFO] Synced LL policy to best_policy: {source} -> {BEST_POLICY_CHECKPOINT}")
        return BEST_POLICY_CHECKPOINT

    if os.path.isfile(BEST_POLICY_CHECKPOINT):
        print(f"[INFO] Using existing best_policy checkpoint: {BEST_POLICY_CHECKPOINT}")
        return BEST_POLICY_CHECKPOINT

    raise FileNotFoundError(
        "No LL policy checkpoint found. Train the LL policy first, pass --checkpoint, "
        f"or place a checkpoint at {BEST_POLICY_CHECKPOINT}."
    )
