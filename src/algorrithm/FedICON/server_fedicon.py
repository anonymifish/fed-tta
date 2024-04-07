import logging
import os

import torch
import wandb

from src.algorrithm.Base.server_base import BaseServer
from src.algorrithm.FedICON.client_fedicon import FedICONClient
import time

class ServerFedICON(BaseServer):
    def __init__(self, device, backbone, configs):
        super(ServerFedICON, self).__init__(device, backbone, configs)
        self.fedavg_rounds = configs.fedavg_rounds
        self.icon_rounds = configs.icon_rounds
        self.finetune_method = configs.finetune_method
        self.finetune_rounds = configs.finetune_rounds

    def fit(self):
        for r in range(self.fedavg_rounds):
            client_weights = []
            client_net_states = []
            client_accuracy = []
            client_train_time = []

            active_clients = self.select_clients()
            for client in active_clients:
                client: FedICONClient
                client_weights.append(len(client.original_dataloader))
                start_time = time.time()
                report = client.fedavg_train()
                end_time = time.time()
                client_net_states.append(report['backbone'])
                client_accuracy.append(report['accuracy'])
                client_train_time.append(end_time - start_time)
                logging.info(f"client{client.cid} training time: {end_time - start_time}")

            global_net_state = self.model_average(client_net_states, client_weights)
            for client in self.clients:
                client.backbone.load_state_dict(global_net_state)
            self.backbone.load_state_dict(global_net_state)

            logging.info(f'average client train time: {sum(client_train_time) / len(client_train_time)}')
            logging.info(
                f'average client accuracy: {sum([client_accuracy[i] * client_weights[i] / sum(client_weights) for i in range(len(active_clients))])}')
            if not self.debug:
                wandb.log({
                    'accuracy': sum([client_accuracy[i] * client_weights[i] / sum(client_weights) for i in
                                     range(len(active_clients))]),
                    'train_time': sum(client_train_time) / len(client_train_time),
                })

            if r % 10 == 0:
                self.make_checkpoint(r)

        for r in range(self.icon_rounds):
            client_weights = []
            client_net_states = []
            client_train_time = []

            active_clients = self.select_clients()
            for client in active_clients:
                client: FedICONClient
                client_weights.append(len(client.train_dataloader))
                start_time = time.time()
                report = client.train()
                end_time = time.time()
                client_net_states.append(report['backbone'])
                client_train_time.append(end_time - start_time)
                logging.info(f"client{client.cid} training time: {end_time - start_time}")

            global_net_state = self.model_average(client_net_states, client_weights)
            for client in self.clients:
                client.backbone.load_state_dict(global_net_state)
            self.backbone.load_state_dict(global_net_state)

            logging.info(f'average client train time: {sum(client_train_time) / len(client_train_time)}')
            if not self.debug:
                wandb.log({
                    'train_time': sum(client_train_time) / len(client_train_time),
                })

            if r % 10 == 0:
                self.make_checkpoint(r)

        for r in range(self.finetune_rounds):
            if self.finetune_method == 'avg':
                client_weights = []
                client_fc_states = []
                active_clients = self.select_clients()
                for client in active_clients:
                    client_weights.append(len(client.original_dataloader))
                    report = client.train()
                    client_fc_states.append(report['fc'])
                global_fc_state = self.model_average(client_fc_states, client_weights)
                for client in self.clients:
                    client.backbone.fc.load_state_dict(global_fc_state)
                self.backbone.fc.load_state_dict(global_fc_state)
            else:
                for client in self.clients:
                    client.finetune()

        self.make_checkpoint(r)

    def make_checkpoint(self, r):
        torch.save(
            {
                'backbone': self.backbone.state_dict(),
                'clients': [client.make_checkpoint for client in self.clients]
            },
            os.path.join(self.checkpoint_path, f'model_round{r}.pt')
        )

    def load_checkpoint(self, checkpoint):
        self.backbone.load_state_dict(checkpoint['backbone'])
        for cid, client_checkpoint in enumerate(checkpoint['clients']):
            self.clients[cid].backbone.load_state_dict(checkpoint['backbone'])
            self.clients[cid].load_checkpoint(client_checkpoint)
