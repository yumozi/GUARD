#!/bin/bash

# Script to pretrain a model, optionally with CURE

RUN_PRETRAIN=false  # Set to false to skip pretraining
DATASET="imagenette"
MODEL="resnet18"
EVAL_MODEL="resnet18"
DATA_ROOT="/home/techt/Desktop/"
EXPERIMENT=24
MODEL_EXP=1
ITERATION=2000
KD_EPOCHS=300
DEBUG=false
IPC=10
GPU=0
CURE=false
LAMDA=100.0
H=3.0
CLIP_GRAD_NORM=0

# Parse command-line arguments. All flags are optional.
# Usage: bash run.sh -d imagenette -x 1 -y 1 -r /home/user/data/ -u 0 -b 0.01 -p -C -h 3.0 -l 100
# -x is the experiment id. Arguments and results will be saved in ./log/{experiment}.json
# If -p is included, it pretrains a model and saves it with the id given by '-y'.
# -y is the id of the teacher model under the (dataset, model) category. Make sure the model exists of '-p' is not set

while getopts ":pd:m:e:x:y:r:i:z:gc:u:Cl:h:b:a:N:" opt; do
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
    N) CLIP_GRAD_NORM="$OPTARG";;
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
            --clip-grad-norm ${CLIP_GRAD_NORM} \
            --resume "./output/${DATASET}_${MODEL}/${MODEL_EXP}/checkpoint.pth"

    fi
    cd ..
fi
