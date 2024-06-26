import os
import time

import torch
import wandb

from src.algorithm.Base.server_base import BaseServer
from src.algorithm.FedTHE.client_fedthe import FedTHEClient
from utils.logger import logger


class FedTHEServer(BaseServer):
    def __init__(self, device, backbone, configs, profiler):
        super(FedTHEServer, self).__init__(device, backbone, configs, profiler)
        self.global_representation = torch.zeros(self.backbone.fc.in_features)

    def fit(self):
        for r in range(self.global_rounds):
            client_weights = []
            client_net_states = []
            client_accuracy = []
            client_train_time = []

            active_clients = self.select_clients()
            self.global_representation = torch.zeros(self.backbone.fc.in_features)
            logger.info(f"fedthe round{r}")
            for client in active_clients:
                client: FedTHEClient
                client_weights.append(len(client.train_dataloader))
                start_time = time.time()
                report = client.train()
                end_time = time.time()
                client_net_states.append(report['backbone'])
                client_accuracy.append(report['accuracy'])
                client_train_time.append(end_time - start_time)
                logger.info(f"client{client.cid} training time: {end_time - start_time}")
                self.global_representation = self.global_representation + client.local_representation / len(
                    active_clients)

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

    def test(self):
        accuracy = []
        client_weights = []
        for client in self.clients:
            client_weights.append(len(client.test_dataloader))
            report = client.test(self.global_representation.detach())
            accuracy.append(report['acc'])

        return sum([accuracy[i] * client_weights[i] / sum(client_weights) for i in range(len(self.clients))])

    def plain_test(self):
        accuracy = []
        client_weights = []
        for client in self.clients:
            client_weights.append(len(client.test_dataloader))
            report = client.plain_test()
            accuracy.append(report['acc'])

        return sum([accuracy[i] * client_weights[i] / sum(client_weights) for i in range(len(self.clients))])

    def make_checkpoint(self, r):
        torch.save(
            {
                'backbone': self.backbone.state_dict(),
                'global_representation': self.global_representation,
                'clients': [client.make_checkpoint() for client in self.clients]
            },
            os.path.join(self.checkpoint_path, f'model_round{r}.pt')
        )

    def load_checkpoint(self, checkpoint):
        self.backbone.load_state_dict(checkpoint['backbone'])
        self.global_representation = checkpoint['global_representation']
        for cid, client_checkpoint in enumerate(checkpoint['clients']):
            self.clients[cid].backbone.load_state_dict(checkpoint['backbone'])
            self.clients[cid].load_checkpoint(client_checkpoint)
