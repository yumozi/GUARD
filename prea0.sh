#!/bin/bash
#SBATCH --job-name="tiny_cure_l_30"
#SBATCH --output="stdout/a0.log"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=64   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcac-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 48:00:00

export OMP_NUM_THREADS=1  # if code is not multithreaded, otherwise set to 8 or 16
module purge
module reset  # load the default Delta modules

source activate adv
bash runfirst.sh -x a0 -y a0 -d tiny-imagenet -r '/scratch/bcac/dataSet/' -p -C -h 3.0 -l 30