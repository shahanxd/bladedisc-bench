#!/bin/bash

echo "===================================================="
echo "Starting Latest PyTorch Environment (v2.3+)"
echo "===================================================="
echo "Workspace mapped to /workspace"

sudo docker run --rm -it --gpus all --ipc=host \
  -v ~/bladedisc-bench:/workspace \
  nvcr.io/nvidia/pytorch:24.02-py3 \
  bash
