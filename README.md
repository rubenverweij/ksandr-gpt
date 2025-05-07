
# Setup environment

Needed shell commands to get VM up and running.

```shell
sudo apt-get update
sudo apt install python3.12-venv
sudo apt install python3-pip
sudo apt install git-all
python3 -m venv venv
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install llama-cpp-python
pip install onprem
sudo apt install code_1.99.3-1744761595_amd64.deb
pip install chromadb langchain_chroma
```




