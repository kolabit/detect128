# Introduction

The point of **detect128** project is create CNN model to detect the 1D barcodes
of type “Code 128”. Model should run on VectorBlox IP-core, running on Microchip
PolarFire FPGA.

Please see the details about VectorBlox here:

- [VectorBlox™ Accelerator: AI/ML Inference for PolarFire® FPGAs and SoCs](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox)

- [VectorBlox™ SDK](https://github.com/Microchip-Vectorblox/VectorBlox-SDK)

This model can be used by smart AI camera, running on Microchip PolarFire FPGA
with VectorBlox IP-core

## Input and Output Data Format

Model input data is RGB image of 416x416 resolution, acquired from videocamera.
Model outputs bounding boxes of detected barcodes.

## Metrics

Classical computer vision detection metrics will are used: mAP@50, mAP@50:95,
Precision, Recall and F2 score.

## Validation

Original dataset split is used (13088/2804/2806). Some test set images are used
for simulation, using VectorBlox simulation utility, according section 6.3 of
VectorBlox user’s manual:
https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/
CoreVectorBlox_IP_Handbook.pdf

## Data

This dataset used it taken
[here](https://www.kaggle.com/datasets/hammadjavaid/barcode-detection-dataset-coco-format)
In includes 18697 images with selected 1d barcodes, acquired in many different
environment. This dataset is single class (barcode). It combines 10+ publicly
available datasets (including Roboflow collections, InventBar, and ParcelBar)

## Modeling

### Baseline

Regular YOLOv8s is be used as a baseline. The plan is to achieve the quality of
model, similar to claimed
[here](https://www.kaggle.com/datasets/hammadjavaid/barcode-detection-dataset-coco-format)
Precision: 0.970, Recall: 0.951, mAP@50: 0.974 YOLOv11,12 are not “officially”
supported by VectorBlox, can't be loaded and run on SoC, so they are not used.
YOLOv9s did not show beter results during experiments

### Main model

Multiple experiments were conducted and YOLOv8s model demonstrated the best
results. YOLOv8s provides acceptable performance (>1FPS) when running on real
hardware. YOLOv11 and YOLOv12, performed great, but they are not supported by
VBX and can't run on the real hardware

### Deployment

The model is converted to VNNX (VBX internal) format. VectorBlox SDK is used for
conversion. Model file with trained weights will be run on the Hardware, based
on PolarFire FPGA with RISC-V CPU and VectorBlox SDK. For example
MPFS250-Video-Kit can be used:
https://www.microchip.com/en-us/development-tool/mpfs250-video-kit

# Setup

## Pre-requisites

Project was tested on Ubuntu 20.04, 22.04, 24.04 with NVidia CUDA 12.4.
To run **Triton Inference Server**, you need to have **docker** installed,
and **NVIDIA Container Toolkit**. Please see:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Installation steps

1. Clone **detect128** repository. Deactivate conda environment if it's running.
2. Install **uv**, and make sure, it installed properly:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

3. Setup Python environment:

```
uv sync
```

4. Activate project virtual environment:

```
source .venv/bin/activate
```

5. Load VectorBlox SDK:

```
./vbxsdk_download.sh
```


# Training

To start the training, run _train.py_ script:

```
python train.py
```

Dataset will be loaded automatically from AWS, if necessary. You will be asked
about _AWS Secret Access Key_. Please contact me directly, regarding the Key.

Datset will be loaded to _data/dataset_ folder. You're welcome to modify _Hydra_
config YAML with training parameters here:

```
conf/train_cfg.yaml
```

**NVidia GPU** is required for training.

Tensorboard server will be started, and browser window will be open
automatically with Tensorboard GUI.

After the finish, the best model will be copied to _data/model_ folder

# Evaluation

For evaluation purposes, you can run model testing with _test.py_ script:

```
python test.py
```

Test datset images will be loaded to _data/dataset/images/test_ folder. Dataset
will be loaded automatically from AWS, if necessary. You will be asked about
_AWS Secret Access Key_. Please contact me directly, regarding the Key. You're
welcome to modify _Hydra_ config YAML with test parameters here:

```
conf/test_cfg.yaml
```

# Production preparation

For production, we need VNNX file (internal VectorBlox IP-core format, based on
ONNX). To create VNNX, run convertion script:

```
./convert.sh
```

Download VNNX model file to MPFS250-Video-Kit, use
_VectorBlox-SDK/example/soc-video-c_ example to test on the PolarFire SoC
hardware.

# Inference

For inference demonstration purposes **Triton Inference Server** is used.
To run **Triton Inference Server**, and **Ultralytics Triton Client**,
run following commands
```
python triton_repo.py
sudo ./run_triton.sh 
python triton_client.py
```

