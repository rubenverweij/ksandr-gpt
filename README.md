# KSANDR GPT

De repository bevat code voor het hosten van taalmodellen voor het bevragen van documenten. Eerst moeten de benodigde software worden geinstaleerd.

1. Docker version 28.1.1
2. Cuda toolkit (wanneer nodig)
3. Git


# Installatie docker omgeving

```shell
# Controleer of de cuda toolkit al aanwezig is
nvidia-smi | grep -P -o "CUDA Version: \d+(\.\d+)+" | grep -P -o "\d+(\.\d+)+"

# Instaleer cuda
lspci | grep -i nvidia

# Controleer de host 
hostnamectl

# Controleer of gcc geinstalleerd is
gcc --version

# Installeer docker
apt install docker.io

# Maak de image
docker build -t ksandr-gpt:0.5 .

# Start de container
docker run -d -p 80:80 ksandr-gpt:0.5 --cap-add SYS_RESOURCE -e USE_MLOCK=0 --gpus=all -v ~/onprem_data:/root/onprem_data
sudo docker run -d -p 80:80 --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 -v ~/onprem_data:/root/onprem_data ksandr-gpt:0.5
docker run -i -t ksandr-gpt:0.5 /bin/bash
```

# Installatie linux host

Opzetten omgeving taalmodel voor Ubuntu 24.04.

```shell
sudo apt-get update
sudo apt install python3.12-venv
sudo apt install python3-pip
sudo apt install git-all
python3 -m venv venv
pip install llama-cpp-python

# GPU route
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Cuda toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9

# Nvidia driver
sudo apt-get install -y nvidia-open

export CUDACXX=/usr/local/cuda-12.9/bin/nvcc
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir 

# Controleer version
sudo apt install nvidia-cuda-toolkit
nvcc --version

# CPU route
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onprem
sudo apt install code_1.99.3-1744761595_amd64.deb
pip install chromadb langchain_chroma
```

# Installatie windows host

```bash 

python3 -m venv venv
source venv/Scripts/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# for pure cpu
pip3 install torch torchvision torchaudio
pip install llama-cpp-python

# Download and install Microsoft C++ Build Tools and make sure Desktop development with C++

# Then install llama-cpp-python for gpu mode: 
set FORCE_CMAKE=1
set CMAKE_ARGS=-DGGML_CUDA=ON
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

# Finally install on prem
pip install onprem
pip install chromadb langchain_chroma
```

