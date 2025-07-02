# NVIDIA CUDA as base image
ARG CUDA_IMAGE
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

WORKDIR /ksandr-gpt
COPY ./ingest /ksandr-gpt/api

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
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

COPY ./requirements.txt /ksandr-gpt/requirements.txt
RUN python3 -m pip install -r /ksandr-gpt/requirements.txt

# Install llama-cpp-python (with CUDA)
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
RUN python3 -m pip install onprem chromadb langchain_chroma 
RUN echo "0 0 * * * python3 /api/ingest_docs.py -vector_db_path /root/onprem_data/vectordb/dense -documents_path /root/ksandr_files/ -chunk_size 400 >> /app/cron.log 2>&1" > /etc/cron.d/ksandr-ingest-cron-job
CMD ["cron", "-f"]