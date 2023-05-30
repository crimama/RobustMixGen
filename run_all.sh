





#? 이거 리스트 다시 정리 -> 엑셀 참고 해서 train 학습 더 돌릴거, 
#? train 이랑 iamge eval 갯수 맞아야 하고 / train + 기존 거 -> text eval과 갯수 맞아야 함 
#? 총 갯수 21개, 기존 14개 추가 실험 7개 
#? git push도 하자 





#train 6개 
train_list="paste_txtshuffle_hardaug_ratio05_4m_fix2 cutmixup_shuffle_hardaug_4m_fix2 cutmixup_shuffle_ratio05_4m_fix2 \
            cutmixup_conjconcat_hardaug_4m_fix2 cutmixup_mixconcat_hardaug_4m_fix2 cutmixup_captioning_hardaug_ratio05_4m_fix2 "

for e in $train_list:
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env Retrieval.py \
        --config ./configs/exp1/$e.yaml --output_dir output/Retrieval_coco_$e \
        --checkpoint ./output/Pretrain/ALBEF_4M.pth  
done 


#Image eval 8개 
img_eval_list="cutmixup_concat_4m_fix2 cutmixup_conjconcat_hardaug_txtaug_ratio05_4m_fix2 paste_txtshuffle_hardaug_ratio05_4m_fix2 cutmixup_shuffle_hardaug_4m_fix2 \
                cutmixup_shuffle_ratio05_4m_fix2 cutmixup_conjconcat_hardaug_4m_fix2 cutmixup_mixconcat_hardaug_4m_fix2 cutmixup_captioning_hardaug_ratio05_4m_fix2 "

for e in $img_eval_list:
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env Evaluate_img.py \
        --output_dir output/Retrieval_coco_$e \
        --checkpoint output/Retrieval_coco_$e/checkpoint_best.pth 
done         


#Text eval 22개 
text_eval_list="baseline_4m  mixgen_4m  paste_txtreplace_backtrans_4m_fix2 paste_txtreplace_hardaug_4m_fix2 \
                cutmixup_hardaug_captioning_4m_fix2 cutmixup_hardaug_textconcat_ratio05_4m_fix2 cutmixup_textconcat_hardaug_4m_fix2 cutmixup_textconcat_ratio05_4m_fix2 \
                paste_captioning_ratio05_4m_fix2 paste_hardaug_captioning_4m_fix2 paste_textconcat_hardaug_4m_fix2 cutmixup_conjconcat_hardaug_ratio05_4m_fix2 \
                cutmixup_txtshuffle_hardaug_ratio05_4m_fix2 cutmixup_mixconcat_hardaug_ratio05_4m_fix2 cutmixup_concat_4m_fix2 cutmixup_conjconcat_hardaug_txtaug_ratio05_4m_fix2 \
                paste_txtshuffle_hardaug_ratio05_4m_fix2 cutmixup_shuffle_hardaug_4m_fix2 cutmixup_shuffle_ratio05_4m_fix2 \
                cutmixup_conjconcat_hardaug_4m_fix2 cutmixup_mixconcat_hardaug_4m_fix2 cutmixup_captioning_hardaug_ratio05_4m_fix2 "

 for e in $text_eval_list:
 do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env Evaluate_txt.py \
        --output_dir output/Retrieval_coco_$e \
        --checkpoint output/Retrieval_coco_$e/checkpoint_best.pth 
done        

