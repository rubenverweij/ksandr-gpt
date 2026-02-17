# ------------------------------------------------------------------------------
# NVIDIA CUDA 13 base for A30 GPU
# ------------------------------------------------------------------------------
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# ------------------------------------------------------------------------------
# System dependencies: build tools, Python, OpenCL, BLAS libraries
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    git \
    build-essential \
    python3 \
    python3-pip \
    gcc \
    wget \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    libclblast-dev \
    libopenblas-dev && \
    # Set up NVIDIA OpenCL vendor
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# ------------------------------------------------------------------------------
# Environment variables for CUDA and llama-cpp
# ------------------------------------------------------------------------------
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS="ignore::UserWarning"

# ------------------------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------------------------
RUN python3 -m pip install --upgrade pip wheel setuptools spacy
RUN python3 -m spacy download nl_core_news_sm
RUN python3 -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# ------------------------------------------------------------------------------
# llama-cpp-python with CUDA enabled
# ------------------------------------------------------------------------------
RUN FORCE_CMAKE=1 CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir llama-cpp-python

# ------------------------------------------------------------------------------
# Other Python packages
# ------------------------------------------------------------------------------
RUN python3 -m pip install --no-cache-dir \
    "langchain>=0.2.14" "langchain-neo4j>=0.1.4" "langchain-community>=0.2.14" \
    PyPDF2 bs4 chromadb langchain_chroma sentence-transformers \
    fastapi[standard] neo4j langchain-text-splitters Levenshtein

# Set working directory and copy application code
# ------------------------------------------------------------------------------
WORKDIR /ksandr-gpt
COPY ./ /ksandr-gpt
