import os
import random

import numpy as np
import torch


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_save_path(configs):
    root_path = os.path.dirname(__file__)
    root_path = os.path.join(os.path.dirname(root_path), "results")
    dataset_setting = f"{configs.dataset}_{configs.num_client}clients_alpha{configs.alpha}"
    task_path = dataset_setting + f"/{configs.method}_{configs.backbone}/{configs.task_name}"
    save_path = os.path.join(root_path, task_path)
    os.makedirs(save_path, exist_ok=True)
    return save_path
