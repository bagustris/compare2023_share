# compare2023_share
Another attempt for ComParE 2023 Emotion Share

Forked from https://github.com/VincentKaras/a-vb-emotions  
See my other attempt: https://github.com/bagustris/compare2023

# Virtual Environment

```
conda create -n compare2023_share python=3.9
conda activate compare2023_share
```

# Installation

```
pip install -r requirements.txt
pip install git+https://github.com/end2you/end2you.git
```

## Running the code (example of use)
For using specific GPU, use `CUDA_VISIBLE_DEVICES` environment variable. For example, to use GPU 1, run the following command:
```
 CUDA_VISIBLE_DEVICES=1 ./train.py -n "window 6" --window_size 6 train --train_dataset_file /data/14_ComParE23_HPC_AIST-SPRT/data/lab/train.csv --val_dataset_file /data/14_ComParE23_HPC_AIST-SPRT/data/lab/devel.csv --test_dataset_file /data/14_ComParE23_HPC_AIST-SPRT/data/lab/test.csv --wav_folder /data/14_ComParE23_HPC_AIST-SPRT/data/wav/
```

See `run_batch.sh` for running in batch mode.

