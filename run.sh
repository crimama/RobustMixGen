#! Pretrain 
#python -m torch.distributed.launch --nproc_per_node=2 --use_env Pretrain.py --config ./configs/Pretrain.yaml 

# #! Image Retrieval - coco 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env Retrieval.py \
#     --config ./configs/Retrieval_coco_mixgen.yaml --output_dir output/Retrieval_coco_mixgen_full \
#     --checkpoint ./output/Pretrain/ALBEF.pth

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env Retrieval.py \
    --config ./configs/Retrieval_coco_small_romix.yaml --output_dir output/Retrieval_coco_small_test \
    --checkpoint ./output/Pretrain/ALBEF.pth

#! Image Retrieval eval - coco 
#  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env Retrieval.py \
#      --config ./configs/Retrieval_coco.yaml --output_dir output/Retrieval_coco \
#      --checkpoint ./output/Retrieval_coco/checkpoint_4.pth --evaluate true 


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
