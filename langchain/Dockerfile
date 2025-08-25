# NVIDIA CUDA as base image
ARG CUDA_IMAGE
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

WORKDIR /ksandr-gpt
COPY ./ /ksandr-gpt/api

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget cron \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Environment variables
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install dependencies
RUN python3 -m pip install --upgrade pip wheel setuptools
RUN python3 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# Install llama-cpp-python (with CUDA)
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
RUN python3 -m pip install redis chromadb langchain_chroma sentence-transformers langchain langchain-huggingface langchain-community fastapi[standard]
CMD ["fastapi", "run", "api/main.py", "--temperature", 0.8, "--source_max", 2, "--score_threshold", 0.6, "--store_type", "sparse", "--port", "8080"]