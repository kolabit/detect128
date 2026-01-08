import shutil
from pathlib import Path


def main():
    model_name = "detect128"
    triton_repo = Path("triton_repo")
    (triton_repo / model_name / "1").mkdir(parents=True, exist_ok=True)

    # Move/copy your exported ONNX into Triton repo
    dst = triton_repo / model_name / "1" / "model.onnx"
    shutil.copy("data/model/detect128.onnx", dst)

    # Create emptyconfig.pbtxt
    (triton_repo / model_name / "config.pbtxt").touch()


if __name__ == "__main__":
    main()
