python generate_soft_label.py \
    --dataset imagenette \
    -a resnet18 \
    -b 1024 \
    -j 8 \
    --ckpt-path ../pretrain/output/imagenette_resnet18/checkpoint.pth \
    --epochs 300 \
    --fkd-seed 42 \
    --input-size 224 \
    --min-scale-crops 0.08 \
    --max-scale-crops 1 \
    --use-fp16 \
    --fkd-path FKD_cutmix_fp16 \
    --mode 'fkd_save' \
    --mix-type 'cutmix' \
    --data ../recover/syn_data/imagenette/rn18_bn0.01_[4K]_x_l2_x_tv.crop_2

