#!/bin/bash

DATASET="imagenette"
MODEL="resnet18"
EXPERIMENT=24
MODEL_EXP=1

# Parse command-line arguments. All flags are optional.
# Usage: bash run.sh -d imagenette -x 1 -y 1 -r /home/user/data/ -u 0 -b 0.01 -p -C -h 3.0 -l 100
# -x is the experiment id. Arguments and results will be saved in ./log/{experiment}.json
# If -p is included, it pretrains a model and saves it with the id given by '-y'.
# -y is the id of the teacher model under the (dataset, model) category. Make sure the model exists of '-p' is not set

while getopts ":d:m:x:y:" opt; do
  case $opt in
    d) DATASET="$OPTARG";;
    m) MODEL="$OPTARG";;
    x) EXPERIMENT="$OPTARG";;
    y) MODEL_EXP="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1;;
  esac
done

# rm -rf "./pretrain/output/${DATASET}_${MODEL}/${MODEL_EXP}"
rm -rf "./recover/syn_data/${DATASET}/${EXPERIMENT}"
rm -rf "./relabel/FKD_cutmix_fp16/${DATASET}/${EXPERIMENT}"
rm -rf "./train/save/final_rn18_fkd/${EXPERIMENT}"
rm -f "./stdout/${EXPERIMENT}.log"
rm -f "./log/${EXPERIMENT}.json"

