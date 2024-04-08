import os
import random

import numpy as np
import torch

from src.algorrithm.FedAvg.client_fedavg import FedAvgClient
from src.algorrithm.FedAvg.server_fedavg import FedAvgServer
from src.algorrithm.FedICON.client_fedicon import FedICONClient
from src.algorrithm.FedICON.server_fedicon import FedICONServer
from src.algorrithm.Method.client_method import MethodClient
from src.algorrithm.Method.server_method import MethodServer


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


def prepare_server_and_clients(backbone, configs):
    device = torch.device(configs.device)
    if configs.method == 'fedavg':
        return FedAvgServer, FedAvgClient
    elif configs.method == 'fedicon':
        return FedICONServer, FedICONClient
    elif configs.method == 'fedthe':
        raise NotImplementedError
    elif configs.method == 'atp':
        raise NotImplementedError
    elif configs.method == 'method':
        return MethodServer, MethodClient
    else:
        raise ValueError(f"method {configs.method} is not supported")
