# RobustMixGen

# Baselin 
ALBEF : https://github.com/salesforce/ALBEF

# Install and Prepare for Training
## Environment 
- docker폴더에 있는 Dockerfile로 Image 설치 후 사용 

또는 
- ufoym/deepo:pytorch 이미지 설치 및 컨테이너 실행 후 `requirement.sh`를 통해 패키지 설치 
## Data 
- https://cocodataset.org/#download 에서 2014COCO 데이터셋 다운 받아 사용 
- 다운 받은 data를 data 폴더에 저장 
- 모든 이미지가 하나의 폴더 내에 있어야 함 
- coco.json에 저장 되어 있는 이미지 디렉토리 변경 
- 이미지 이름은 고정 하고 directory만 바꿔주면 됨 

# 실행 
```
python -m torch.distributed.launch --nproc_per_node=2 --use_env Pretrain.py --config ./configs/Pretrain.yaml 
```

