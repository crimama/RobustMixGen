# RobustMixGen

# Baseline 
ALBEF : https://github.com/salesforce/ALBEF

# Install and Prepare for Training
## Environment 
- docker폴더에 있는 Dockerfile로 Image 설치 후 사용 

또는 
- ufoym/deepo:pytorch 이미지 설치 및 컨테이너 실행 후 `requirement.sh`를 통해 패키지 설치 
## Data 
- MSCOCO
```
python ./data/download_coco.py
```

# 실행 
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env Pretrain.py --config ./configs/Pretrain.yaml 
```

