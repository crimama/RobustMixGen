#! Pretrain 
#python -m torch.distributed.launch --nproc_per_node=2 --use_env Pretrain.py --config ./configs/Pretrain.yaml 

# #! Image Retrieval - coco 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
    --config ./configs/Retrieval_coco.yaml --output_dir output/Retrieval_coco \
    --checkpoint ./output/Pretrain/ALBEF.pth

#! VQA 
# python -m torch.distributed.launch --nproc_per_node=2 --use_env VQA.py \
#     --config ./configs/VQA.yaml \
#     --output_dir output/VQA \
#     --checkpoint ./output/Pretrain/ALBEF.pth

#!  SNLI-VE : Visual Entailment 
#python -m torch.distributed.launch --nproc_per_node=2 --use_env VE.py \
#    --config ./configs/VE.yaml \
#    --output_dir output/VE \
#   --checkpoint ./output/Pretrain/ALBEF.pth

#! NLVR
# python -m torch.distributed.launch --nproc_per_node=2 --use_env Pretrain_nlvr.py \
#     --config ./configs/NLVR_pretrain.yaml \
#     --output_dir output/NLVR_pretrain \
#     --checkpoint ./output/Pretrain/ALBEF.pth

#! Visual Grounding 
# python -m torch.distributed.launch --nproc_per_node=2 --use_env Pretrain_nlvr.py \
#     --config ./configs/NLVR_pretrain.yaml \
#     --output_dir output/NLVR_pretrain \
#     --checkpoint ./output/Pretrain/ALBEF.pth
