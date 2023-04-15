# Dockerfile
FROM ufoym/deepo:pytorch

# 패키지 업데이트 및 설치
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0 \
    git -y \
    && rm -rf /var/lib/apt/lists/*

RUN /usr/bin/python3.8 -m pip install --upgrade pip
RUN pip install cython
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install transformers==4.8.1
RUN pip install timm==0.4.9
RUN pip install wget
RUN pip install opencv-python
RUN pip install ruamel.yaml
RUN pip install salesforce-lavis


#COCO Download