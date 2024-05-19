import os
import time

import torch
import torch.nn.functional as F
import wandb
from torch import autograd

from src.algorithm.Base.server_base import BaseServer
from src.algorithm.FedIIR.client_fediir import FedIIRClient
from utils.logger import logger


class FedIIRServer(BaseServer):
    def __init__(self, device, backbone, configs, profiler):
        super().__init__(device, backbone, configs, profiler)

        params = self.backbone.fc.parameters()
        self.grad_mean = tuple(torch.zeros_like(p).to(self.device) for p in params)
        self.ema = configs.fediir_ema

    def fit(self):
        for r in range(self.global_rounds):
            if self.use_profile:
                self.profiler.step()
            client_weights = []
            client_net_states = []
            client_accuracy = []
            client_train_time = []

            active_clients = self.select_clients()
            self.grad_mean = self.mean_grad(active_clients)
            logger.info(f"round{r}")
            for client in active_clients:
                client: FedIIRClient
                client_weights.append(len(client.train_dataloader))
                start_time = time.time()
                report = client.train()
                end_time = time.time()
                client_net_states.append(report['backbone'])
                client_accuracy.append(report['accuracy'])
                client_train_time.append(end_time - start_time)
                logger.info(f"client{client.cid} training time: {end_time - start_time}")

            global_net_state = self.model_average(client_net_states, client_weights)
            for client in self.clients:
                client.backbone.load_state_dict(global_net_state)
            self.backbone.load_state_dict(global_net_state)

            logger.info(f'average client train time: {sum(client_train_time) / len(client_train_time)}')
            logger.info(
                f'average client accuracy: {sum([client_accuracy[i] * client_weights[i] / sum(client_weights) for i in range(len(active_clients))])}')
            if not self.debug:
                wandb.log({
                    'accuracy': sum([client_accuracy[i] * client_weights[i] / sum(client_weights) for i in
                                     range(len(active_clients))]),
                    'train_time': sum(client_train_time) / len(client_train_time),
                })

            if (r + 1) % 10 == 0:
                self.make_checkpoint(r + 1)

    def mean_grad(self, active_clients):
        total_batch = 0
        grad_sum = tuple(torch.zeros_like(g).to(self.device) for g in self.grad_mean)
        self.backbone.to(self.device)
        for client in active_clients:
            for data, target in client.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.backbone(data)

                loss = F.cross_entropy(logits, target)
                grad_batch = autograd.grad(loss, self.backbone.fc.parameters(), create_graph=False)

                grad_sum = tuple(g1 + g2 for g1, g2 in zip(grad_sum, grad_batch))
                total_batch += 1

        grad_mean_new = tuple(grad / total_batch for grad in grad_sum)
        self.backbone.cpu()
        return tuple(self.ema * g1 + (1 - self.ema) * g2 for g1, g2 in zip(self.grad_mean, grad_mean_new))

    def plain_test(self):
        return self.test()

    def test(self):
        accuracy = []
        client_weights = []
        for client in self.clients:
            client_weights.append(len(client.test_dataloader))
            report = client.test()
            accuracy.append(report['acc'])

        return sum([accuracy[i] * client_weights[i] / sum(client_weights) for i in range(len(self.clients))])

    def make_checkpoint(self, r):
        torch.save(self.backbone.state_dict(), os.path.join(self.checkpoint_path, f'model_round{r}.pt'))

    def load_checkpoint(self, checkpoint):
        self.backbone.load_state_dict(checkpoint)
        for client in self.clients:
            client.backbone.load_state_dict(checkpoint)
