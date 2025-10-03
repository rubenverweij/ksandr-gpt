# KSANDR GPT

De repository bevat code voor het hosten van taalmodellen voor het bevragen van documenten. Eerst moeten de benodigde software worden geinstaleerd.

1. Docker version 28.1.1
2. Cuda toolkit (wanneer nodig)
3. Git


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

# Maak de api image and start een container
docker build -t ksandr-gpt:0.XX .
docker run --network host -d --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 -v /home/ubuntu/onprem_data:/root/onprem_data -v /home/ubuntu/ksandr_files:/root/ksandr_files ksandr-gpt-langchain:0.34

# Maak de ingest image en start de container
docker build -t ksandr-ingest:0.XX .
docker run --network host -d --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 -v /home/ubuntu/onprem_data:/root/onprem_data -v /home/ubuntu/ksandr_files:/root/ksandr_files ksandr-gpt-ingest:0.1 

# Kopieer aantal documenten
docker cp docs/txt/ <container name>:/root/ksandr_files/
```

Vervolgens kunnen documenten worden geupload:

```shell
python3 api/ingest_docs.py -path /root/ksandr_files/

curl -X POST http://localhost:8080/ask_stream \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Welke nederlandse netbeheerders hebben een LK ELA12 schakelinstallatie?",
   "permission": {
    "aads": {
      "cat-1": [10535]
    }
  }
}'

curl -X POST http://localhost:8080/ask_stream \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Wat zijn ageing assets?",
  "permission": {
    "aads": {
      "cat-1": [2061]
    },
    "documents": [1, 2],
    "groups": [1, 2],
    "ese": true,
    "esg": false,
    "rmd": [1, 2],
    "dga": [1, 2]
  }
}'

curl -X POST http://localhost:8080/ask \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Wat weet je van de testopstelling klassiek en Lamke sonde"
}'


curl -X POST http://localhost:8080/summarise \
-H "Content-Type: application/json" \
-d '{
  "doc_path": "/root/ksandr_files/aads/2061/cat-1/documents/13866.txt",
  "concept": "Geregistreerde storingen"
}'


watch -n 0.5 nvidia-smi
```

# Troubleshooting

1. Wanneer `docker build -t ksandr-gpt:0.XX .` niet lukt omdat packages niet gevonden kunnen worden kan een herstart van docker noodzakelijk zijn: `sudo service docker restart`
2. Wanneer een container geen response geeft `docker stop <container_id>`. Het `container_id` is te vinden met `docker ps`. Daarna `docker start` of start een nieuwe container `docker run --network host -d --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 -v /home/ubuntu/onprem_data:/root/onprem_data -v /home/ubuntu/ksandr_files:/root/ksandr_files ksandr-gpt:<versie>`

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
pip install llama-cpp-python

# Download en installeer Microsoft C++ Build Tools
set FORCE_CMAKE=1
set CMAKE_ARGS=-DGGML_CUDA=ON
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

# Installeer onprem
pip install onprem
pip install chromadb langchain_chroma
```

