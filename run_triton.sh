#!/bin/bash
#
#python triton_repo.py

docker pull nvcr.io/nvidia/tritonserver:24.09-py3

docker run -d --rm \
  --runtime=nvidia --gpus 0 \
  -v $PWD/triton_repo:/models \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  nvcr.io/nvidia/tritonserver:24.09-py3 \
  tritonserver --model-repository=/models


