#!/bin/bash

cp $HOME/projects/def-ravanelm/data/parkinson/PREPARE_WavLM.zip $SLURM_TMPDIR
cd $SLURM_TMPDIR
unzip PREPARE_WavLM.zip

cd $HOME/parkinson/ParkinsonSpeechAI/recipes/PREPARE/detection

