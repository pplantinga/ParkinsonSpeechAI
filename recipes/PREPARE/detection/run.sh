#!/bin/bash

echo "Script name: $0"
for arg in "$@"; do
    echo "- $arg"
done

module load StdEnv/2020
module load python/3.10.2
source $HOME/myenv/bin/activate

scp $HOME/projects/def-ravanelm/data/parkinson/PREPARE_WavLM.zip $SLURM_TMPDIR
cd $SLURM_TMPDIR
unzip PREPARE_WavLM.zip
cd $HOME/parkinson/ParkinsonSpeechAI/recipes/PREPARE/detection

for i in $(seq $1 $2)
do
    torchrun --nproc_per_node=4 train.py hparams/wavlm_pre_ecapa.yaml \
             --data_folder=$SLURM_TMPDIR/PREPARE --results_dir=$HOME/scratch/results/prepare \
             --seed=$i --experiment_name=$3 ${@:4} &
    wait
done


