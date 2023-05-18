#!/usr/bin/env python3
# script to calculate spearmann correlation from experiment results
# example
# python calculate_spearmann.py results/2021-05-25_14-05-01/
import sys
import pandas as pd
import numpy as np

from scipy.stats import spearmanr

def spearmann(dir_pred):
    """
    calculate spearmann correlation between prediction and ground truth
    """
    pred = pd.read_csv(dir_pred + "predictions/val/high.csv")
    # true label, change with your own path here
    true = pd.read_csv('/data/14_ComParE23_HPC_AIST-SPRT/data/lab/devel.csv')

    stats = []

    for i in pred.columns[1:]:
        stats.append(spearmanr(pred[i], true[i])[0])
    
    return np.mean(stats)

if __name__ == "__main__":
    dir_pred = sys.argv[1]
    print(spearmann(dir_pred))