import os
import random

import numpy as np
import torch

from src.algorithm.ATP.client_atp import ATPClient
from src.algorithm.ATP.server_atp import ATPServer
from src.algorithm.FedAvg.client_fedavg import FedAvgClient
from src.algorithm.FedAvg.server_fedavg import FedAvgServer
from src.algorithm.FedICON.client_fedicon import FedICONClient
from src.algorithm.FedICON.server_fedicon import FedICONServer
from src.algorithm.FedTHE.client_fedthe import FedTHEClient
from src.algorithm.FedTHE.server_fedthe import FedTHEServer
from src.algorithm.Method.client_method import MethodClient
from src.algorithm.Method.server_method import MethodServer


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_save_path(configs):
    root_path = os.path.dirname(__file__)
    root_path = os.path.join(os.path.dirname(root_path), "results")
    if configs.dataset in ['cifar10', 'cifar100']:
        dataset_setting = f"{configs.dataset}_{configs.num_client}clients_alpha{configs.alpha}"
    else:
        dataset_setting = f"{configs.dataset}_leave-{configs.leave_one_out}"
    task_path = dataset_setting + f"/{configs.method}_{configs.backbone}/{configs.task_name}"
    save_path = os.path.join(root_path, task_path)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def prepare_server_and_clients(configs):
    if configs.method == 'fedavg':
        return FedAvgServer, FedAvgClient
    elif configs.method == 'fedicon':
        return FedICONServer, FedICONClient
    elif configs.method == 'fedthe':
        return FedTHEServer, FedTHEClient
    elif configs.method == 'atp':
        return ATPServer, ATPClient
    elif configs.method == 'method':
        return MethodServer, MethodClient
    else:
        raise ValueError(f"method {configs.method} is not supported")
