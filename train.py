from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import subprocess
import time
import webbrowser
from omegaconf import DictConfig, OmegaConf
#import getpass!!!
from ultralytics import YOLO

RUNS_DIR = "runs"  # default filder for logging
TB_PORT = 6006


def data_load():
    """
    Prompt for AWS credentials (if not provided), set them in env,
    and run `dvc pull` in the given project directory.

    Credentials are only set for the subprocess, not globally.
    """
    # Ask AWS Access Key ID
    aws_access_key_id = "AKIAWXZSCPTRZPABNF5X" #input("AWS Access Key ID: ").strip()
    # AWS Secret Access Key:
    aws_secret_access_key = input("AWS Secret Access Key for KeyID=AKIAWXZSCPTRZPABNF5X: ").strip()
    #!!! aws_secret_access_key = getpass.getpass("AWS Secret Access Key: ").strip() !!!
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS credentials must not be empty")

    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    env["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    return subprocess.run(
        ["dvc", "pull"],
        cwd=str("."), #cwd=str("../")
        env=env,
        text=True,
        capture_output=True,
       )

def start_tensorboard(logdir=RUNS_DIR, port=TB_PORT):
    # Start TensorBoard in background
    tb_proc = subprocess.Popen(
        ["tensorboard", f"--logdir={logdir}", f"--port={port}", "--bind_all"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Sleep to give TensorBoard time to start
    time.sleep(3)

    # Open browser
    webbrowser.open(f"http://localhost:{port}/#timeseries")

    return tb_proc


# Optional extra TB logging
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def _as_float(x: Any) -> Optional[float]:
    """Convert common metric types to float safely."""
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


class ExtraTensorBoardLogger:
    """
    Adds extra TensorBoard scalars using Ultralytics callbacks.

    Ultralytics already writes TensorBoard event files.
    This logger is useful when you want custom tags or additional metrics.
    """

    def __init__(self, log_dir: Path):
        if SummaryWriter is None:
            raise RuntimeError(
                "tensorboard SummaryWriter not available. "
                "Install with: uv add tensorboard (and ensure torch is installed)."
            )
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def close(self):
        self.writer.flush()
        self.writer.close()

    def on_train_epoch_end(self, trainer) -> None:
        # trainer.metrics typically includes training loss components in recent Ultralytics versions
        # We'll log whatever we can find.
        epoch = getattr(trainer, "epoch", None)
        if epoch is None:
            return

        metrics: Dict[str, Any] = {}
        # Try multiple known locations because Ultralytics internals can vary slightly by version
        if hasattr(trainer, "metrics") and isinstance(trainer.metrics, dict):
            metrics.update(trainer.metrics)
        if hasattr(trainer, "tloss"):
            # sometimes tloss is a tensor/list, but we'll attempt to log it
            metrics["train/tloss"] = trainer.tloss

        for k, v in metrics.items():
            fv = _as_float(v)
            if fv is None:
                continue
            # Tag namespace to avoid collisions with Ultralytics TB tags
            tag = f"extra/{k}"
            self.writer.add_scalar(tag, fv, epoch)

        self.writer.flush()

    def on_fit_epoch_end(self, trainer) -> None:
        # Validation metrics may be present in trainer.metrics or trainer.validator.metrics
        epoch = getattr(trainer, "epoch", None)
        if epoch is None:
            return

        val_metrics: Dict[str, Any] = {}

        if hasattr(trainer, "validator") and trainer.validator is not None:
            vm = getattr(trainer.validator, "metrics", None)
            # vm can be a dict-like or custom object. Try dict first.
            if isinstance(vm, dict):
                val_metrics.update(vm)
            else:
                # Some versions expose a results_dict
                rd = getattr(vm, "results_dict", None)
                if isinstance(rd, dict):
                    val_metrics.update(rd)

        # Also merge trainer.metrics if it contains val/ keys
        if hasattr(trainer, "metrics") and isinstance(trainer.metrics, dict):
            for k, v in trainer.metrics.items():
                if (
                    "val" in k.lower()
                    or "map" in k.lower()
                    or "precision" in k.lower()
                    or "recall" in k.lower()
                ):
                    val_metrics[k] = v

        for k, v in val_metrics.items():
            fv = _as_float(v)
            if fv is None:
                continue
            self.writer.add_scalar(f"extra/val/{k}", fv, epoch)

        self.writer.flush()


@hydra.main(version_base=None, config_path="conf", config_name="train_cfg")
def main(cfg: DictConfig) -> None:
    
    print("Load the dataset from AWS S3\n")
    dl_res = data_load()
    print(dl_res.stdout)
    print(dl_res.stderr)
    if dl_res.returncode != 0:
        print("Dataset load failed\n")
        return

    print("Hydra config:\n", OmegaConf.to_yaml(cfg))

    # Resolve paths relative to original working directory (Hydra changes cwd)
    orig_cwd = Path(hydra.utils.get_original_cwd())
    data_yaml = (orig_cwd / cfg.train.data).resolve()
    model_name_or_path = cfg.train.model  # can be built-in weights name or local path

    # Ultralytics output directory:
    # runs/<project>/<name>/
    project_dir = (orig_cwd / cfg.logging.project).resolve()
    run_name = str(cfg.logging.name)

    # Create YOLO model
    model = YOLO(model_name_or_path)

    extra_logger: Optional[ExtraTensorBoardLogger] = None
    callbacks_added = False

    # Train arguments (Ultralytics)
    train_args = dict(
        data=str(data_yaml),
        epochs=int(cfg.train.epochs),
        imgsz=int(cfg.train.imgsz),
        batch=int(cfg.train.batch),
        device=str(cfg.train.device),
        workers=int(cfg.train.workers),
        optimizer=str(cfg.train.optimizer),
        lr0=float(cfg.train.lr0),
        lrf=float(cfg.train.lrf),
        weight_decay=float(cfg.train.weight_decay),
        patience=int(cfg.train.patience),
        seed=int(cfg.train.seed),
        pretrained=bool(cfg.train.pretrained),
        project=str(project_dir),
        name=run_name,
        exist_ok=bool(cfg.logging.exist_ok),
        # TensorBoard logs are produced automatically by Ultralytics in the run dir.
        # You can also add verbose=True/False as needed:
        verbose=True,
    )

    # Extra TB logging (optional)
    if bool(cfg.logging.extra_tensorboard):
        # Put extra TB logs inside the same run folder for convenience:
        # runs/<project>/<name>/<extra_tb_subdir>/
        extra_tb_dir = project_dir / run_name / str(cfg.logging.extra_tb_subdir)
        extra_tb_dir.mkdir(parents=True, exist_ok=True)
        extra_logger = ExtraTensorBoardLogger(log_dir=extra_tb_dir)

        # Add callbacks to Ultralytics trainer lifecycle
        # These callback names are supported by Ultralytics' callback system.
        # If you are on an older/newer version and a hook name differs,
        # you can adjust to the ones printed by model.callbacks keys.
        model.add_callback("on_train_epoch_end", extra_logger.on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", extra_logger.on_fit_epoch_end)
        callbacks_added = True

    try:
        tb_process = start_tensorboard()
        results = model.train(**train_args)
        # results is a Ultralytics Results object for final val (varies)
        print("Training completed.")
        print(results)
        input("Press Enter to close Tensorboard GUI")
        tb_process.terminate()
    finally:
        if extra_logger is not None:
            extra_logger.close()
        if callbacks_added:
            # Not strictly necessary; keeps state clean if you reuse the model object
            pass


if __name__ == "__main__":
    main()
