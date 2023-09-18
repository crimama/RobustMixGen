



CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --default_setting ./configs/Pretrain.yaml \
num_workers 32 \
wandb.wandb_use False


