



CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --default_setting ./configs/default.yaml --task_setting ./configs/ve.yaml \
batch_size_train 64 \
romixgen.base.romixgen_true True \
romixgen.image.hard_aug True \
mixgen False \
exp_name add-hard-aug