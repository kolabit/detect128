# test.py

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO


def data_load():
    """
    Prompt for AWS credentials (if not provided), set them in env,
    and run `dvc pull` in the given project directory.

    Credentials are only set for the subprocess, not globally.
    """
    # Ask AWS Access Key ID
    aws_access_key_id = "AKIAWXZSCPTRZPABNF5X"  # input("AWS Access Key ID: ")
    # AWS Secret Access Key:
    aws_secret_access_key = input(
        "AWS Secret Access Key for KeyID=AKIAWXZSCPTRZPABNF5X: "
    ).strip()
    # !!! aws_secret_access_key=getpass.getpass("AWS Secret Access Key:")
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS credentials must not be empty")

    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    env["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    return subprocess.run(
        ["dvc", "pull", "--force"],
        cwd=str("."),  # cwd=os.getcwd(),
        env=env,
        text=True,
        # capture_output=True,
    )


def f_beta(p: float, r: float, beta: float = 2.0) -> float:
    if p <= 0.0 and r <= 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * p + r
    return (1.0 + b2) * p * r / denom if denom > 0 else 0.0


def extract_metrics(val_result: Any) -> Dict[str, Optional[float]]:
    def to_float(x) -> Optional[float]:
        try:
            return float(x)
        except Exception:
            return None

    box = getattr(val_result, "box", None)
    if box is not None:
        p = to_float(getattr(box, "mp", None))
        r = to_float(getattr(box, "mr", None))
        map50 = to_float(getattr(box, "map50", None))
        map5095 = to_float(getattr(box, "map", None))
    else:
        p = to_float(getattr(val_result, "mp", None))
        r = to_float(getattr(val_result, "mr", None))
        map50 = to_float(getattr(val_result, "map50", None))
        map5095 = to_float(getattr(val_result, "map", None))

    f2 = f_beta(p, r, beta=2.0) if (p is not None and r is not None) else None

    return {
        "precision": p,
        "recall": r,
        "f2": f2,
        "map50": map50,
        "map50_95": map5095,
    }


# Enable config from conf/test_cfg.yaml
@hydra.main(version_base=None, config_path="conf", config_name="test_cfg")
def main(cfg: DictConfig) -> None:
    # if bool(cfg.get("print_config", False)):
    # print(OmegaConf.to_yaml(cfg))
    # return
    print("Load the dataset from AWS S3\n")
    dl_res = data_load()
    # print(dl_res.stdout)
    # print(dl_res.stderr)
    if dl_res.returncode != 0:
        print("Dataset load failed\n")
        return

    print("Hydra config:\n", OmegaConf.to_yaml(cfg))

    weights = str(cfg.get("weights", "detect128.pt"))
    data = str(cfg.get("data", "data.yaml"))
    split = str(cfg.get("split", "test"))

    data_yaml = Path(data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"YAML not found: {data_yaml.resolve()}")

    val_kwargs: Dict[str, Any] = {
        "data": data,
        "split": split,
        "imgsz": cfg.get("imgsz", 640),
        "batch": cfg.get("batch", 16),
        "conf": cfg.get("conf", 0.001),
        "iou": cfg.get("iou", 0.7),
        "max_det": cfg.get("max_det", 300),
        "half": cfg.get("half", False),
        "rect": cfg.get("rect", False),
        "verbose": cfg.get("verbose", True),
        "project": cfg.get("project", "runs/eval"),
        "name": cfg.get("name", "exp"),
        "exist_ok": cfg.get("exist_ok", False),  # !!!
        "save_json": cfg.get("save_json", False),
        "plots": cfg.get("plots", True),
    }

    if cfg.get("device"):
        val_kwargs["device"] = str(cfg.device)

    if "extra" in cfg:
        extra = OmegaConf.to_container(cfg.extra, resolve=True)
        if isinstance(extra, dict):
            val_kwargs.update(extra)

    model = YOLO(weights)
    res = model.val(**val_kwargs)

    m = extract_metrics(res)

    print("\nDetect128 quality metrics on test data:")

    print(
        f"Precision: {m['precision']:.4f}"
        if m["precision"] is not None
        else "Precision:   n/a"
    )

    print(
        f"Recall:    {m['recall']:.4f}"
        if m["recall"] is not None
        else "Recall:      n/a"
    )

    print(f"F2:  {m['f2']:.4f}")

    print(f"mAP@50: {m['map50']:.4f}")

    print(f"mAP@50:95: {m['map50_95']:.4f}")

    out_dir = Path(val_kwargs["project"]) / val_kwargs["name"]
    print(f"\nArtifacts directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
