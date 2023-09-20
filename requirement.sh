pip install --upgrade pip 
#pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install transformers==4.26.1
pip install timm==0.4.9 wget ruamel.yaml salesforce-lavis

# git clone https://github.com/cocodataset/cocoapi.git
# cd cocoapi/PythonAPI
# make

pip install albumentations
pip install omegaconf
pip install wandb 
pip install easydict

# Image perturbation 
pip install wand

# Text perturbation 
pip install nltk 
pip install nlpaug==1.1.11
pip install sentencepiece

#CV2 
apt-get install libglib2.0
apt-get update && apt-get install libgl1
apt-get install libmagickwand-dev
pip install opencv-python==4.8.0.74

#json -> for SNLI-VE  
pip install jsonlines 

#detectrion2
## for dsba4
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

## for dsba6 