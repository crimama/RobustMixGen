#python -m torch.distributed.launch --nproc_per_node=2 --use_env Pretrain.py --config ./configs/Pretrain.yaml 
python -m torch.distributed.launch --nproc_per_node=2 --use_env Retrieval.py \
    --config ./configs/Retrieval_coco.yaml --output_dir output/Retrieval_coco \
    --checkpoint ./output/Pretrain/ALBEF.pth
