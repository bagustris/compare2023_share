"""
Optimizer code
"""

from statistics import mode
import torch
import torch.nn as nn
from end2you.utils import Params
from model.models import AbstractModel
from losses import Criterion
import bitsandbytes as bnb

def get_optimizer(train_params:Params, model:AbstractModel, criterion:Criterion) -> torch.optim.Optimizer:

    opt_name = str(train_params.optimizer).lower()

    opt_parameters =    [
                {"params": model.feature_extractor.parameters(),
                    "lr": train_params.fe_lr
                },
                {
                    "params": model.classifier.parameters(),
                }
            ]
    if len(list(criterion.parameters())) > 0:   # add parameters of the criterion to be optimized
        opt_parameters.append({"params": criterion.parameters()})

    if opt_name == "sgd":

        optimizer = torch.optim.SGD(
            opt_parameters,
            #model.parameters(),
            momentum=0.9,
            lr=train_params.lr
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            #model.parameters(),
            opt_parameters,
            lr=train_params.lr
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            #.parameters(),
            opt_parameters,
            lr=train_params.lr,
            weight_decay=train_params.weight_decay
        )
    elif opt_name == "adam8bit":
        optimizer = bnb.optim.Adam8bit(
            #.parameters(),
            opt_parameters,
            lr=train_params.lr,
            weight_decay=train_params.weight_decay,
            betas=(0.9, 0.995),
            min_8bit_size=16384
        )

    elif opt_name == "adamw8bit":
        optimizer = bnb.optim.AdamW8bit(
            #.parameters(),
            opt_parameters,
            lr=train_params.lr,
            weight_decay=train_params.weight_decay,
            betas=(0.9, 0.995),
            min_8bit_size=16384
        )

    elif opt_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            #model.parameters(),
            opt_parameters,
            lr=train_params.lr,
            alpha=0.95,
            eps=1e-7
        )

    else:
        raise NotImplementedError

    return optimizer

def get_scheduler(optimizer:torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:

    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        mode="max",
        factor=0.5,  # based on paper, halved every 5 epochs
        patience=5
    )