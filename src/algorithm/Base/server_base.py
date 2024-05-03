import copy
import random

import torch


class BaseServer:
    def __init__(self, device, backbone, configs, profiler):
        self.clients = []
        self.join_raio = configs.join_ratio
        self.global_rounds = configs.global_rounds
        self.checkpoint_path = configs.checkpoint_path
        self.backbone = copy.deepcopy(backbone)
        self.device = device
        self.debug = configs.debug
        self.profiler = profiler

    def select_clients(self):
        if self.join_raio == 1.0:
            return self.clients
        else:
            return random.sample(self.clients, int(round(len(self.clients) * self.join_raio)))

    @staticmethod
    def model_average(client_net_states, client_weights):
        state_avg = copy.deepcopy(client_net_states[0])
        client_weights = [w / sum(client_weights) for w in client_weights]

        for k in state_avg.keys():
            state_avg[k] = torch.zeros_like(state_avg[k])
            for i, w in enumerate(client_weights):
                state_avg[k] = state_avg[k] + client_net_states[i][k] * w

        return state_avg
