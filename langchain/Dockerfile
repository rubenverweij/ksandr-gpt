# NVIDIA cuda als basis image
ARG CUDA_IMAGE
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Omgevingsvariabelen
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Installeer dependencies
RUN python3 -m pip install --upgrade pip wheel setuptools spacy
RUN python3 -m spacy download nl_core_news_sm
RUN python3 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# Installeer llama-cpp-python (met cuda)
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
RUN python3 -m pip install bs4 redis chromadb langchain_chroma sentence-transformers langchain langchain-huggingface langchain-community fastapi[standard] neo4j neo2py

WORKDIR /ksandr-gpt
COPY ./ /ksandr-gpt/api

CMD ["fastapi", "run", "api/main.py", "--port", "8080"]