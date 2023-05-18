import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import CCC
from end2you.utils import Params
from typing import List

from dataset import EMOTIONS, TYPES


"""
Loss definitions
"""

class CCC_Loss(nn.Module):
    """
    Computes CCC loss for multidimensional targets and averages / 
    weighs the result
    """

    def __init__(self, num_targets: int, weight=None) -> None:
        super().__init__()

        self.num_targets = num_targets

        if weight is not None:
            self.weight == nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = None

    def forward(self, preds, targets):

        assert preds.shape == targets.shape

        if preds.dim() > 2:
            preds = preds.view(-1, preds.shape[-1])
            targets = targets.view(-1, targets.shape[-1])

        # compute CCC
        ccc = CCC(preds, targets)
        ccc_losses = 1 - ccc

        return torch.mean(ccc_losses) 


class MSE_loss(nn.Module):
    """
    Computes MSE loss for multidimensional targets and averages / 
    weighs the result
    """

    def __init__(self, num_targets: int, weight=None) -> None:
        super().__init__()

        self.num_targets = num_targets
        self.losses = nn.MSELoss()

    def forward(self, preds, targets, return_all=True):
    
        assert preds.shape == targets.shape

        if preds.dim() > 2:
            preds = preds.view(-1, preds.shape[-1])
            targets = targets.view(-1, targets.shape[-1])

        # compute MSE
        mse_losses = self.losses(preds, targets)

        return torch.mean(mse_losses)
 

class Criterion(nn.Module):
    """
    criterion definition for training. 
    """
    def __init__(self, params: Params) -> None:
        super().__init__()
        
        self.tasks = ["high"]  # Add other task for future
        self.tasks_dict = {
            "high": {
                "type": "regression",
                "dimensions": EMOTIONS,
            },
        }
        self.params = params
        loss_strategy = params.train.loss_strategy

        # create loss modules
        loss_dict = {}
        for t in self.tasks:
            if loss_strategy == "mse":
                print("Using MSE loss")
                loss_dict[t] = MSE_loss(num_targets=len(self.tasks_dict[t]["dimensions"]))
            else:
                print("Using CCC loss")
                loss_dict[t] = CCC_Loss(num_targets=len(self.tasks_dict[t]["dimensions"]))
        # put losses in a module dict so they will be registered properly
        self.losses = nn.ModuleDict(loss_dict)


    def forward(self, preds, targets, return_all=True):
        """
        Basic additive loss. Returns mean of the losses, plus optional a dict of individual task results
        """
        loss = 0.0
        task_losses = []

        for task_index, task in enumerate(self.tasks):

            if isinstance(preds, dict):
                pred = preds[task]
            else:
                pred = preds[task_index]
            if isinstance(targets, dict):
                target = targets[task]
            else:
                target = targets[task]

            #task_loss = self.losses[task](pred, target)
            task_losses.append(self.losses[task](pred, target))

        #if isinstance(pred, dict) and isinstance(target, dict):
        loss = torch.mean(torch.stack(task_losses))

        if return_all:
            return loss, {t: task_losses[i].item() for i, t in enumerate(self.tasks)}

        else: 
            return loss


def criterion_factory(params: Params) -> Criterion:
    """
    factory helper that generates the desired criterion based on the 
    loss_strategy flag
    :params Params object, containing train.loss_strategy str: ccc or mse
    """ 

    loss_strategy = params.train.loss_strategy

    if loss_strategy == "ccc" or loss_strategy == "mse":
        print("Using CCC losses")
        return Criterion(params=params)
    else: 
        raise NotImplementedError("{} not implemented".format(loss_strategy))
