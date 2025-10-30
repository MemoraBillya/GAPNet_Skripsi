"""Utility helper to launch GAPNet training on Kaggle GPUs.

The script mirrors the manual steps you would normally run inside a Kaggle
Notebook and adds inline comments so every action is explicit.  By default it
expects that you have uploaded a dataset to Kaggle and mounted it under the
standard `/kaggle/input/<dataset-name>` folder.
"""

import argparse
import os
import subprocess
from pathlib import Path
from typing import List

# Default Kaggle mount points that stay constant across sessions.
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORK_ROOT = Path("/kaggle/working")


def _ensure_symlink(src: Path, dest: Path) -> Path:
    """Create a symbolic link so GAPNet sees the dataset under `./data/`.

    Kaggle notebooks have read-only access to `/kaggle/input`, while
    `/kaggle/working` is writable.  GAPNet's training script expects the
    training data to live underneath `./data/`, so we expose the uploaded
    dataset through a symlink inside the repo root.
    """

    if dest.exists():
        # Reuse the existing directory (either a folder or a symlink).
        return dest

    if not src.exists():
        raise FileNotFoundError(
            f"Dataset directory {src} does not exist. Check the Kaggle dataset "
            "name or ensure it is added to the notebook."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Symlinks are cheap and avoid copying potentially large datasets.
    os.symlink(src, dest, target_is_directory=True)
    return dest


def _build_train_command(repo_root: Path, cli_args: argparse.Namespace) -> List[str]:
    """Translate high-level Kaggle options into `scripts/train.py` arguments."""

    train_script = repo_root / "scripts" / "train.py"

    cmd: List[str] = [
        "python",
        str(train_script),
        "--data_dir",
        str(cli_args.data_dir),
        "--arch",
        cli_args.arch,
        "--max_epochs",
        str(cli_args.max_epochs),
        "--batch_size",
        str(cli_args.batch_size),
        "--num_workers",
        str(cli_args.num_workers),
        "--lr",
        str(cli_args.lr),
        "--lr_mode",
        cli_args.lr_mode,
        "--ms",
        str(int(cli_args.multi_scale)),
        "--bcedice",
        str(int(cli_args.use_bce_dice)),
        "--adam_beta2",
        str(cli_args.adam_beta2),
        "--group_lr",
        str(int(cli_args.group_lr)),
        "--igi",
        str(int(cli_args.ignore_index)),
        "--supervision",
        str(cli_args.supervision),
        "--gpu",
        str(cli_args.use_gpu),
        "--gpu_id",
        cli_args.gpu_id,
        "--savedir",
        str(cli_args.save_dir),
    ]

    if cli_args.resume is not None:
        cmd.extend(["--resume", str(cli_args.resume)])

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAPNet training on Kaggle.")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help=(
            "Name of the Kaggle dataset as it appears under /kaggle/input. "
            "For example, if files are under /kaggle/input/gapnet-sod, pass "
            "'gapnet-sod'."
        ),
    )
    # Every option below accepts both dash and underscore styles so users can
    # copy either CLI form from the README or their own notebooks without
    # editing flag names.
    parser.add_argument(
        "--arch",
        default="convnextv2_atto",
        help="Backbone identifier passed to scripts/train.py (e.g., mobilenetv2).",
    )
    parser.add_argument(
        "--max-epochs",
        "--max_epochs",
        type=int,
        default=40,
        help="Number of training epochs to run on Kaggle.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=8,
        help="Mini-batch size per iteration. Adjust to fit Kaggle GPU memory.",
    )
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        type=int,
        default=4,
        help="How many DataLoader workers to spawn.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.5e-4,
        help="Initial learning rate (passed through to scripts/train.py).",
    )
    parser.add_argument(
        "--lr-mode",
        "--lr_mode",
        default="poly",
        choices=["poly", "step"],
        help="Learning-rate schedule to reuse from scripts/train.py.",
    )
    parser.add_argument(
        "--multi-scale",
        "--multi_scale",
        action="store_true",
        help="Enable multi-scale augmentation (maps to --ms 1).",
    )
    parser.add_argument(
        "--use-bce-dice",
        "--use_bce_dice",
        action="store_true",
        help="Train with the BCE+Dice loss combination (maps to --bcedice 1).",
    )
    parser.add_argument(
        "--adam-beta2",
        "--adam_beta2",
        type=float,
        default=0.99,
        help="beta2 hyper-parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--group-lr",
        "--group_lr",
        action="store_true",
        help="Toggle grouped learning rate for backbone parameters.",
    )
    parser.add_argument(
        "--ignore-index",
        "--ignore_index",
        action="store_true",
        help="Enable ignore-index handling in the CE+O loss (maps to --igi 1).",
    )
    parser.add_argument(
        "--supervision",
        type=int,
        default=8,
        help="Supervision mode integer mirrored from scripts/train.py.",
    )
    parser.add_argument(
        "--use-gpu",
        "--use_gpu",
        default=True,
        type=lambda value: str(value).lower() == "true",
        help="Whether to ask scripts/train.py to use CUDA (default: True).",
    )
    parser.add_argument(
        "--gpu-id",
        "--gpu_id",
        default="0",
        help="CUDA device index visible inside the Kaggle session.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint path (inside /kaggle/input) to resume from.",
    )
    parser.add_argument(
        "--run-name",
        default="kaggle",
        help="Suffix appended to the savedir for organizing multiple runs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    data_src = KAGGLE_INPUT_ROOT / args.dataset_name
    data_dest = repo_root / "data"
    prepared_data_dir = _ensure_symlink(data_src, data_dest)
    print(
        f"Dataset root mounted: {prepared_data_dir} -> {data_src}."
        " Contents should include DUTS-TR/, SOD/, etc."
    )

    # Store checkpoints and logs under /kaggle/working so they survive until the
    # notebook finishes (they can be downloaded as output artifacts).
    save_dir = (KAGGLE_WORK_ROOT / "gapnet_runs" / args.run_name).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Enrich the CLI args namespace with derived paths used by the command
    # builder so we have a single source of truth.
    args.data_dir = prepared_data_dir
    args.save_dir = str(save_dir) + "/"
    args.resume = args.resume

    # Set CUDA visibility explicitly. Kaggle exposes a single GPU at index 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    train_cmd = _build_train_command(repo_root, args)
    print("Launching training command:\n", " ".join(train_cmd))

    # Ensure the repository root is on PYTHONPATH so `import models` works even
    # when scripts/train.py is executed via an absolute path.  Kaggle notebooks
    # often modify sys.path implicitly, so we make the requirement explicit here.
    env = os.environ.copy()
    python_path_entries = [str(repo_root)]
    if env.get("PYTHONPATH"):
        python_path_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path_entries)

    subprocess.run(train_cmd, check=True, cwd=repo_root, env=env)


if __name__ == "__main__":
    main()
