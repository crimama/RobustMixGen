# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# # pip install transformers==4.26.1
# # pip install timm==0.4.9
# # pip install opencv-python
# # pip install ruamel.yaml 
# pip install transformers==4.8.1 timm==0.4.9 wget opencv-python ruamel.yaml salesforce-lavis

# apt-get install libglib2.0

# git clone https://github.com/cocodataset/cocoapi.git
# cd cocoapi/PythonAPI
# make



pip install --upgrade pip 
pip install ruamel.yaml
pip install transformers==4.26.1
pip install wandb 
# Image perturbation 
pip install wand
# Text perturbation 
pip install nltk 
pip install nlpaug==1.1.11
pip install sentencepiece
#CV2 
apt-get update && apt-get install libgl1
apt-get install libmagickwand-dev
#json -> for SNLI-VE  
pip install jsonlines 