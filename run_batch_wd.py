#!/usr/bin/env bash

# This script runs the batch job for the set of features

# features="wav2vec2-base wavlm-base-plus unispeech-sat-base-plus hubert-base-ls960 wav2vec2-large wavlm-large unispeech-sat-large hubert-large-ll60k wav2vec2-xls-r-300m"

feature="hubert-large-ll60k"
wds="0.002 0.003 0.004 0.005 0.006 0.007 0.009 0.01"

for wd in $wds; do
    echo "Running batch job for weight decay = $wd"
    ./train.py --name "experiment_$feature" --feature_extractor $feature --batch_size 8 train --train_dataset_file /data/14_ComParE23_HPC_AIST-SPRT/data/lab/train.csv --val_dataset_file /data/14_ComParE23_HPC_AIST-SPRT/data/lab/devel.csv --test_dataset_file /data/14_ComParE23_HPC_AIST-SPRT/data/lab/test.csv --wav_folder /data/14_ComParE23_HPC_AIST-SPRT/data/wav/ --weight_decay $wd
done


