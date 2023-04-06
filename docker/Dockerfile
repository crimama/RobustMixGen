FROM ufoym/deepo:pytorch

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install transformers==4.8.1
RUN pip install timm==0.4.9
RUN pip install opencv-python
RUN pip install ruamel.yaml 
RUN apt-get update && apt-get install libgl1
RUN apt-get install libglib2.0