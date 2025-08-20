# NVIDIA CUDA as base image
ARG CUDA_IMAGE
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

WORKDIR /ksandr-gpt
COPY ./ingest /ksandr-gpt/api

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget cron \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Omgevingsvariabelen
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Installeer dependencies
RUN python3 -m pip install --upgrade pip wheel setuptools
RUN python3 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
RUN python3 -m pip install --upgrade pytest cmake scikit-build 

# Install llama-cpp-python (with CUDA)
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
RUN python3 -m pip install onprem chromadb langchain_chroma 
RUN echo "0 0 * * * python3 /ksandr-gpt/api/ingest_docs.py -vector_db_path /root/onprem_data/vectordb/sparse -documents_path /root/ksandr_files/ -chunk_size 300 -chunk_overlap 100 >> /tmp/cron.log 2>&1" > /etc/cron.d/ksandr-ingest-cron-job
RUN chmod 0644 /etc/cron.d/ksandr-ingest-cron-job
RUN crontab /etc/cron.d/ksandr-ingest-cron-job
CMD ["cron", "-f"]