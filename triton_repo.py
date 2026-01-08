from pathlib import Path
import shutil

def main():
    model_name = "detect128"
    triton_repo = Path("triton_repo")
    (triton_repo / model_name / "1").mkdir(parents=True, exist_ok=True)

    # Move/copy your exported ONNX into Triton repo
    shutil.copy("data/model/detect128.onnx", triton_repo / model_name / "1" / "model.onnx")

    # Create emptyconfig.pbtxt
    (triton_repo / model_name / "config.pbtxt").touch()

if __name__ == "__main__":
    main()