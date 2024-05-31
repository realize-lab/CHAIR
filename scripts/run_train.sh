#!/bin/sh

script_name=$1
job_name=$2
seed=$3
dataset=$4
train_mode=$5

sbatch --partition=spgpu --gpus=1 --cpus-per-gpu=8 --nodes=1 --mem-per-cpu=11500m --time=00-48:00:00 --job-name=$job_name --output="newlogs/$job_name.out" $script_name $seed $dataset $train_mode $6