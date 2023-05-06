pip install nlpaug 
pip install --upgrade huggingface_hub
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.8.1
pip install timm==0.4.9
pip install opencv-python
pip install ruamel.yaml 
apt-get update && apt-get install libgl1
apt-get install libglib2.0

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make

pip install sentencepiece