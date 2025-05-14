
# Setup environment

Needed shell commands to get VM up and running.

```shell
sudo apt-get update
sudo apt install python3.12-venv
sudo apt install python3-pip
sudo apt install git-all
python3 -m venv venv
pip install llama-cpp-python

# for gpu route
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

## cuda toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2404-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9

# nvidia driver
sudo apt-get install -y nvidia-open


export CUDACXX=/usr/local/cuda-12.9/bin/nvcc
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir 

# check version
sudo apt install nvidia-cuda-toolkit
nvcc --version


# for cpu route
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


pip install onprem
sudo apt install code_1.99.3-1744761595_amd64.deb
pip install chromadb langchain_chroma
```




