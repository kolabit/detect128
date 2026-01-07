#!/bin/bash
#
# 1) Get VectorBlox-SDK, release-v2.0.3
#
bash ./vbxsdk_download.sh
#
# 2) Switch to VBX environment
#
source ./VectorBlox-SDK/setup_vars.sh
#
# 3) Convert model
#

echo "Checking and activating VBX Python Environment..."
if [ -z $VBX_SDK ]; then
    echo "\$VBX_SDK not set. Please run 'source setup_vars.sh' from the SDK's root folder" && exit 1
fi
source $VBX_SDK/vbx_env/bin/activate

echo "STEP 1: Export to tflite int8"
PRJ_DIR=$PWD
cd $PRJ_DIR/data/model
yolo export model=detect128.pt format=tflite int8 || true
cp detect128_saved_model/detect128_full_integer_quant.tflite detect128.tflite

if [ -f detect128.tflite ]; then
   echo "STEP 2: Cutting graph"
   tflite_cut detect128.tflite -c 190 197 207 214 224 231
   mv detect128.0.tflite detect128.cut.tflite
fi

if [ -f detect128.cut.tflite ]; then
   tflite_preprocess detect128.cut.tflite  --scale 255
fi

if [ -f detect128.cut.pre.tflite ]; then
    echo "STEP 3: Generating VNNX for V1000 configuration..."
    vnnx_compile -c V1000 -t detect128.cut.pre.tflite -o detect128.vnnx
fi

if [ -f detect128.vnnx ]; then
    echo "Running Simulation..."
    python $VBX_SDK/example/python/yoloInfer.py detect128.vnnx $PRJ_DIR/data/dataset/images/test/barcode_detector_test_025565.jpg -v 8 -l $PRJ_DIR/data/dataset/barcode.names
fi

deactivate
