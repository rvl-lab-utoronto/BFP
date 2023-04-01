# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import torch
import numpy as np

def get_device() -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BASE_PATH = './data/'

def set_base_path(path:str) -> None:
    global BASE_PATH
    BASE_PATH = path
    if not BASE_PATH.endswith("/"):
        BASE_PATH += "/"

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return BASE_PATH


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
