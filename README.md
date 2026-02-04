# KSANDR GPT

De repository bevat code voor het hosten van taalmodellen voor het bevragen van documenten. Eerst moeten de benodigde software worden geinstaleerd.

1. Docker version 28.2.2, build 28.2.2-0ubuntu1~22.04.1
2. Cuda toolkit (wanneer nodig)
3. Git, vervolgens repo clonen in `/home/ubuntu/`

## Installatie dependencies

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
```

- Voor het starten van de LLM container met API: `langchain\README.md`
- Voor het starten van de Neo4j container t.b.v. graph database: `langchain\graphdb\README.md`
- Voor het ingesten van nieuwe documenten: `langchain\README.md`
- Voor het ingesten van de neo4j data: `langchain\graphdb\README.md`

## Troubleshooting

1. Wanneer `docker build -t ksandr-gpt-langchain:0.XX .` niet lukt omdat packages niet gevonden kunnen worden kan een herstart van docker noodzakelijk zijn: `sudo service docker restart`
2. Wanneer een container geen response geeft `docker stop <container_id>`. Het `container_id` is te vinden met `docker ps`. Daarna `docker start` of start een nieuwe container `docker run --network host -d --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 -v /home/ubuntu/onprem_data:/root/onprem_data -v /home/ubuntu/ksandr_files:/root/ksandr_files ksandr-gpt:<versie>`
3. Wanneer de kernel driver een update nodig heeft? `nvidia-smi` geeft in dat geval een version mismatch. Oplossing: `sudo apt install --reinstall nvidia-driver-580` en vervolgens `sudo reboot`

Vervolgens kunnen documenten worden geupload:

```shell
Voorbeeld requests:

# Ophalen metadata
curl -X GET http://localhost:8080/metadata

# Ophalen status request
curl -X GET http://localhost:8080/status/f846e504-9731-4a08-b9f2-29f13a2d1329

# Model herstarten met andere context
curl -X POST http://localhost:8080/set-context \
  -H "Content-Type: application/json" \
  -d '{
    "n_ctx": "2048"
  }'

# samenvatting maken
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/root/ksandr_files/aads/10540/cat-1/documents/13834.txt",
    "summary_length": 600
  }'


# Voorbeeld context vraag
curl -X POST http://localhost:8080/context \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<|im_start|>system\nJe bent een behulpzame assistent. Geef alleen bondig antwoord op de laatstgestelde vraag.\n<|im_end|>\n<|im_start|>user\nHallo, wat is magnefix md?\n<|im_end|>\n<|im_start|>assistant\nDat is een schakelinstallatie.\n<|im_end|>\n<|im_start|>user\nDoor welke fabrikant is die gemaakt?\n<|im_end|>\n<|im_start|>assistant\nEaton.\n<|im_end|>\n<|im_start|>user\nWat weet je van hen?\n<|im_end|>\n<|im_start|>assistant"
  }'


curl -X POST http://localhost:8080/ask \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Geef de faalvormen van de DB10?"
}'


curl -X POST http://localhost:8080/ask \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Geef de populatiegegevens per netbeheerder van de XIria",
  "permission": {
    "aads": {
      "cat-1": [
        318, 655, 1555, 1556, 1557, 1558, 2059, 2061, 2963,
        8825, 8827, 9026, 9027, 9028, 10535, 10536, 10540,
        10542, 10545, 10546, 10547, 10548, 10551, 10552,
        10553, 10554, 10555, 10556, 10557
      ],
      "cat-2": [10884]
    }
  }
}'


curl -X POST http://localhost:8080/ask \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Geef een overzicht van de AAD dossiers?",
  "permission": {
    "aads": {
      "cat-1": [
        318, 655, 1555, 1556, 1557, 1558, 2059, 2061, 2963,
        8825, 8827, 9026, 9027, 9028, 10535, 10536, 10540,
        10542, 10545, 10546, 10547, 10548, 10551, 10552,
        10553, 10554, 10555, 10556, 10557
      ]
    },
    "documents": [
      10733, 10734, 10735, 10736, 10737, 10738, 10739,
      10740, 10759, 10760, 10761, 10863, 10864, 10865,
      10866, 10867, 10868, 10869, 10999, 11041, 11265,
      11274, 11275, 11383, 11384, 11385, 11386, 11551,
      12403, 12404, 12405, 12406, 12462, 12463, 12556,
      13682, 13683, 13684, 13685, 13686, 13687, 13692,
      13697, 13698, 14025, 14198, 14199, 14215, 14221,
      14247, 14371, 14373, 14374, 14375, 14376, 14377,
      14378, 14379, 14380, 14383, 14385, 14399
    ],
    "groups": [
      260, 277, 278, 280, 281, 826, 827, 828, 832, 1217,
      1218, 1961, 1968, 2175, 2408, 9001, 9358, 9359,
      10193, 10541, 10678, 10684, 10685, 10686, 10687,
      10688, 10689, 10690, 10691, 10692, 10693, 10694,
      10695, 10696, 10697, 10698, 10699, 10700, 10702,
      10703, 10705, 10706, 10707, 10708, 10709, 10710,
      10712, 10713
    ],
    "ese": true,
    "esg": true,
    "rmd": [
      14284172, 18613440, 18860943, 22990584, 27200298,
      29964311, 35966728, 37781047, 44972089, 48539960,
      49958430, 51044944, 62155242, 62276376, 63665801,
      65056104, 65542778, 70474078, 74132964, 83602093,
      89942749, 91846034, 92430633, 98606898
    ],
    "dga": [
      14284172, 18613440, 18860943, 22990584, 27200298,
      29964311, 35966728, 37781047, 44972089, 48539960,
      49958430, 51044944, 62155242, 62276376, 63665801,
      65056104, 65542778, 70474078, 74132964, 83602093,
      89942749, 91846034, 92430633, 98606898
    ]
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

## Installatie linux host

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

