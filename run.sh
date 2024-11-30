#!/bin/bash

RUN_PRETRAIN=false  # Set to false to skip pretraining
DATASET="imagenette"
MODEL="resnet18"
EVAL_MODEL="resnet18"
DATA_ROOT="/media/techt/DATA/data/"
EXPERIMENT=24
MODEL_EXP=1
ITERATION=2000
KD_EPOCHS=300
DEBUG=false
IPC=10
GPU=0
CURE=false
GR=false
LAMDA=100.0
H=3.0
CDA=False
BETA=1e-4
ADV="none"

# Parse command-line arguments. All flags are optional.
# Usage: bash run.sh -d imagenette -x 1 -y 1 -r /home/user/data/ -u 0 -b 0.01 -p -C -h 3.0 -l 100
# -x is the experiment id. Arguments and results will be saved in ./log/{experiment}.json
# If -p is included, it pretrains a model and saves it with the id given by '-y'.
# -y is the id of the teacher model under the (dataset, model) category. Make sure the model exists of '-p' is not set

while getopts ":pd:m:e:x:y:r:i:z:gc:u:Cl:h:b:a:AGB:v:" opt; do
  case $opt in
    p) RUN_PRETRAIN=true;;
    d) DATASET="$OPTARG";;
    m) MODEL="$OPTARG";;
    e) EVAL_MODEL="$OPTARG";;
    x) EXPERIMENT="$OPTARG";;
    y) MODEL_EXP="$OPTARG";;
    r) DATA_ROOT="$OPTARG";;
    i) ITERATION="$OPTARG";;
    z) KD_EPOCHS="$OPTARG";;
    g) DEBUG=true;;
    c) IPC="$OPTARG";;
    u) GPU="$OPTARG";;
    C) CURE=true;;
    l) LAMDA="$OPTARG";;
    h) H="$OPTARG";;
    b) R_BN="$OPTARG";;
    a) LR="$OPTARG";;
    A) CDA=true;;
    G) GR=true;;
    B) BETA="$OPTARG";;
    v) ADV="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

# Test if the code can run
if [ "$DEBUG" = true ]; then
    ITERATION=2
    KD_EPOCHS=2
    IPC=2
fi

if [[ "${DATASET}" == "tiny-imagenet" ]]; then
    # Only set R_BN and LR if they weren't provided as command-line arguments
    : ${R_BN:=1.0}
    : ${LR:=0.1}
else
    : ${R_BN:=0.01}
    : ${LR:=0.25}
fi

# Handle the case when the dataset has 10 classes and IPC is small, the effective batchsize in the relableling step is
# smaller, causing error in the FKD step.
if [[ "${DATASET}" != "tiny-imagenet" && "${DATASET}" != "imagenet" && "${IPC}" -lt 10 ]]; then
    KD_BATCH_SIZE=$((10 * IPC))
else
    KD_BATCH_SIZE=100
fi

DATA_PATH="${DATA_ROOT}${DATASET}"

if [ "${RUN_PRETRAIN}" = true ]; then
    cd ./pretrain/
    if [[ "${DATASET}" == cifar* ]]; then
        CUDA_VISIBLE_DEVICES=${GPU} python pretrain_cifar.py \
            --dataset ${DATASET} \
            --model ${MODEL} \
            --data-path ${DATA_PATH} \
            --exp-name ${MODEL_EXP} \
            --epochs 200 \
            --cure ${CURE} \
            --lamda ${LAMDA} \
            --h ${H}
    else
        CUDA_VISIBLE_DEVICES=${GPU} python pretrain.py \
            --dataset ${DATASET} \
            --model ${MODEL} \
            --data-path ${DATA_PATH} \
            --exp-name ${MODEL_EXP} \
            --opt sgd \
            --lr 0.025 \
            --wd 1e-4 \
            --batch-size 32 \
            --lr-scheduler cosine \
            --epochs 50 \
            --augmix-severity 0 \
            --ra-magnitude 0 \
            --cure ${CURE} \
            --lamda ${LAMDA} \
            --h ${H} \
            --gr ${GR} \
            --beta ${BETA} \
            --adv ${ADV}

    fi
    cd ..
fi

cd ./recover/
if [[ "${DATASET}" == cifar* ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python recover_cifar.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --ckpt-path "../pretrain/output/${DATASET}_${MODEL}/${MODEL_EXP}/checkpoint.pth" \
    --exp-name ${EXPERIMENT} \
    --ipc ${IPC} \
    --batch-size 100 \
    --lr 0.25 \
    --iteration 1000 \
    --r-bn 0.01 \
    --cda ${CDA} \
    --store-best-images
else
    CUDA_VISIBLE_DEVICES=${GPU} python data_synthesis.py \
            --dataset ${DATASET} \
            --model ${MODEL} \
            --ckpt-path "../pretrain/output/${DATASET}_${MODEL}/${MODEL_EXP}/checkpoint.pth" \
            --real-data-path ${DATA_PATH} \
            --exp-name ${EXPERIMENT} \
            --ipc ${IPC} \
            --batch-size 100 \
            --lr ${LR} \
            --iteration ${ITERATION} \
            --l2-scale 0 \
            --tv-l2 0 \
            --r-bn ${R_BN} \
            --cda ${CDA} \
            --verifier \
            --store-best-images

fi

cd ..

cd ./relabel/
if [[ "${DATASET}" == cifar* ]]; then
    CUDA_VISIBLE_DEVICES=${GPU} python relabel_cifar.py \
    --epochs 400 \
    --dataset ${DATASET} \
    --exp-name ${EXPERIMENT} \
    --output-dir "../train/save/final_rn18_fkd/${EXPERIMENT}/" \
    --syn-data-path "../recover/syn_data/${DATASET}/${EXPERIMENT}" \
    --real-data-path ${DATA_PATH} \
    --teacher-path "../pretrain/output/${DATASET}_${MODEL}/${MODEL_EXP}/checkpoint.pth" \
    --ipc ${IPC} --batch-size 128
else
    CUDA_VISIBLE_DEVICES=${GPU} python generate_soft_label.py \
        --dataset ${DATASET} \
        --model ${MODEL} \
        --exp-name ${EXPERIMENT} \
        --ckpt-path "../pretrain/output/${DATASET}_${MODEL}/${MODEL_EXP}/checkpoint.pth" \
        --batch-size ${KD_BATCH_SIZE} \
        --epochs ${KD_EPOCHS} \
        --workers 8 \
        --fkd-seed 42 \
        --input-size 224 \
        --min-scale-crops 0.08 \
        --max-scale-crops 1 \
        --use-fp16 \
        --fkd-path FKD_cutmix_fp16 \
        --mode 'fkd_save' \
        --mix-type 'cutmix' \
        --data "../recover/syn_data/${DATASET}/${EXPERIMENT}"
fi
cd ..


if [[ "${DATASET}" != cifar* ]]; then
    cd ./train/
    CUDA_VISIBLE_DEVICES=${GPU} python train_FKD.py \
        --dataset ${DATASET} \
        --model ${EVAL_MODEL} \
        --batch-size ${KD_BATCH_SIZE} \
        --epochs ${KD_EPOCHS} \
        --exp-name ${EXPERIMENT} \
        --cos \
        --temperature 20 \
        --workers 8 \
        --gradient-accumulation-steps 1 \
        --train-dir "../recover/syn_data/${DATASET}/${EXPERIMENT}" \
        --val-dir ${DATA_PATH}/val \
        --fkd-path "../relabel/FKD_cutmix_fp16/${DATASET}/${EXPERIMENT}" \
        --mix-type 'cutmix' \
        --output-dir "./save/final_rn18_fkd/${EXPERIMENT}/"
    cd ..
fi