#!/bin/bash

echo "===================================================="
echo "Starting BladeDISC Environment (PyTorch 2.0.1)"
echo "===================================================="
echo "Workspace mapped to /workspace"

sudo docker run --rm -it --gpus all \
  -v ~/bladedisc-bench:/workspace \
  bladedisc/bladedisc:latest-runtime-torch-2.0.1-cu118 \
  python3 workspace/ex1_bladedisc_fusion.py
