# INTRODUCTION

The point of **detect128** project is create CNN model to detect the 1D barcodes of type “Code 128”.
Model should run on VectorBlox IP-core, running on Microchip PolarFire FPGA.

Please see the details about VectorBlox here:

- [VectorBlox™ Accelerator: AI/ML Inference for PolarFire® FPGAs and SoCs](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox)

- [VectorBlox™ SDK](https://github.com/Microchip-Vectorblox/VectorBlox-SDK)

This model can be used by smart AI camera, running on Microchip PolarFire FPGA with VectorBlox IP-core


## Input and Output Data Format
Model input data is RGB image of 416x416 resolution, acquired from videocamera.
Model outputs bounding boxes of detected barcodes.

## Metrics
Classical computer vision detection metrics will are used: mAP, IoU, Precision, Recall and F1 score.

## Validation
Since I have large dataset, “classical” 80/10/10 way split will be used. Test set will be also used for
simulation, using VectorBlox simulation utility, according section 6.3 of VectorBlox user’s manual:
https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/
CoreVectorBlox_IP_Handbook.pdf
VectorBlox SDK will be used in dedicated docker container.

## Data
This dataset used it taken [here](https://www.kaggle.com/datasets/hammadjavaid/barcode-detection-dataset-coco-format)
In includes 18697 images with selected 1d barcodes, acquired in many different environment. This
dataset is single class (barcode). It combines 10+ publicly available datasets (including Roboflow
collections, InventBar, and ParcelBar)

## Modeling

### Baseline
Regular YOLOv8s or YOLOv8n model will be used as a baseline. The plan is to achieve the quality of
model, similar to claimed [here](https://www.kaggle.com/datasets/hammadjavaid/barcode-detection-dataset-coco-format)
Precision: 0.970, Recall: 0.951, mAP@50: 0.974
YOLOv11,12 are not “officially” supported by VectorBlox, so appropriate research should be conducted.

### Main model
Main model should provide acceptable performance running on real hardware.
Experiments with Nano and Small versions of YOLOv8 will be conducted.
I expect to achieve 100ms detection time.
I will  try YOLOv11 and YOLOv12, because according to the VectorBlox Manual, it supports TensorFlow
Light models, and since YOLOv11 and YOLOv12 are supported by TFLight, so they should be
supported by VBX. I will use Ultralytics package for the model training in the appropriate dedicated
docker container.

### Deployment
Model should be converted either to TFLight or VNNX (VBX internal) format. The best one will be
selected, basing on model speed and quality. VectorBlox SDK will be used in docker container for
conversion. Model file with trained weights will be run on the Hardware, based on PolarFire FPGA with
RISC-V CPU and VectorBlox SDK.
For example MPFS250-Video-Kit can be used:
https://www.microchip.com/en-us/development-tool/mpfs250-video-kit
