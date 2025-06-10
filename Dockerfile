ARG CUDA_IMAGE
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST=0.0.0.0

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

# Install dependencies
RUN python3 -m pip install --upgrade pip wheel setuptools
RUN python3 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
RUN python3 -m pip install --upgrade pytest cmake scikit-build fastapi uvicorn \
  sse-starlette pydantic-settings starlette-context

# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python

# Install Streamlit
# RUN python3 -m pip install streamlit

RUN python3 -m pip install --upgrade onprem chromadb langchain_chroma fastapi[standard]

WORKDIR /ksandr-gpt
COPY ./api /ksandr-gpt/api

# Default to launching into the REPL
# CMD python3

CMD ["fastapi", "run", "api/main.py", "--port", "80"]