# ================================
# Base image: NVIDIA CUDA 13.0 for A30 GPU, Ubuntu 22.04
# ================================
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

# ================================
# Install system dependencies: compilers, Python, OpenCL, BLAS, and utilities
# ================================
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
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# ================================
# Set environment variables for CUDA, llama-cpp, and Python behavior
# ================================
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONWARNINGS="ignore::UserWarning"

# ================================
# Install core Python tooling and spaCy Dutch language model
# ================================
RUN python3 -m pip install --upgrade pip wheel setuptools spacy
RUN python3 -m spacy download nl_core_news_sm

# ================================
# Install PyTorch suite with CUDA 13.0 compatibility
# ================================
RUN python3 -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# ================================
# Install llama-cpp-python compiled with CUDA support
# ================================
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python

# ================================
# Install additional Python packages for LangChain and document/graph/LLM support
# ================================
RUN python3 -m pip install "langchain>=0.2.14" "langchain-neo4j>=0.1.4" "langchain-community>=0.2.14"
RUN python3 -m pip install PyPDF2 bs4 chromadb langchain_chroma sentence-transformers langchain-huggingface fastapi[standard] neo4j langchain-text-splitters Levenshtein

# ================================
# Copy application code into container and set working directory
# ================================
WORKDIR /ksandr-gpt
COPY ./ /ksandr-gpt
