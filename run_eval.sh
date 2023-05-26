# weight_list="romixgen_textconcat_hardaug_4m_fix2" 

# for w in $weight_list:
# do
#     CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --use_env Evaluate.py \
#         --output_dir output/Retrieval_coco_$w \
#         --checkpoint output/Retrieval_coco_$w/checkpoint_best.pth 
# done 

#! img 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env Evaluate_img.py \
    --output_dir output/Retrieval_coco_cutmixup_conjconcat_hardaug_txtaug_ratio05_4m_fix2 \
    --checkpoint output/Retrieval_coco_cutmixup_conjconcat_hardaug_txtaug_ratio05_4m_fix2/checkpoint_best.pth 

#! txt 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env Evaluate_txt.py \
    --output_dir output/Retrieval_coco_cutmixup_conjconcat_hardaug_txtaug_ratio05_4m_fix2 \
    --checkpoint output/Retrieval_coco_cutmixup_conjconcat_hardaug_txtaug_ratio05_4m_fix2/checkpoint_best.pth     
