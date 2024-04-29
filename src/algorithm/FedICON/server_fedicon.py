import logging
import os
import time

import torch
import wandb

from src.algorithm.Base.server_base import BaseServer
from src.algorithm.FedICON.client_fedicon import FedICONClient
from utils.logger import logger


class FedICONServer(BaseServer):
    def __init__(self, device, backbone, configs):
        super(FedICONServer, self).__init__(device, backbone, configs)
        self.icon_rounds = configs.icon_rounds
        self.finetune_method = configs.finetune_method
        self.finetune_rounds = configs.finetune_rounds

    def fit(self):
        for r in range(self.icon_rounds):
            client_weights = []
            client_net_states = []
            client_train_time = []
            client_epoch_loss = []

            active_clients = self.select_clients()
            logger.info(f"icon round{r}")
            for client in active_clients:
                client: FedICONClient
                client_weights.append(len(client.train_dataloader))
                start_time = time.time()
                report = client.train()
                end_time = time.time()
                client_net_states.append(report['backbone'])
                client_epoch_loss.append(report['epoch_loss'])
                client_train_time.append(end_time - start_time)
                logger.info(f"client{client.cid} training time: {end_time - start_time}")

            global_net_state = self.model_average(client_net_states, client_weights)
            for client in self.clients:
                client.backbone.load_state_dict(global_net_state)
            self.backbone.load_state_dict(global_net_state)

            logger.info(f'average client train time: {sum(client_train_time) / len(client_train_time)}')
            if not self.debug:
                wandb.log({
                    'train_time': sum(client_train_time) / len(client_train_time),
                    'loss': sum(client_epoch_loss) / len(client_epoch_loss),
                })

            if (r + 1) % 5 == 0:
                self.make_checkpoint(r + 1)

        for r in range(self.finetune_rounds):
            if self.finetune_method == 'avg':
                client_weights = []
                client_fc_states = []
                active_clients = self.select_clients()
                logger.info(f"finetune round{r}")
                for client in active_clients:
                    client_weights.append(len(client.original_dataloader))
                    report = client.finetune()
                    client_fc_states.append(report['fc'])
                global_fc_state = self.model_average(client_fc_states, client_weights)
                for client in self.clients:
                    logger.info(f"finetune client{client.cid}")
                    client.backbone.fc.load_state_dict(global_fc_state)
                self.backbone.fc.load_state_dict(global_fc_state)
            else:
                for client in self.clients:
                    client.finetune()

        self.make_checkpoint(self.icon_rounds + self.finetune_rounds)

    def test(self):
        accuracy = []
        client_weights = []
        for client in self.clients:
            client_weights.append(len(client.test_dataloader))
            report = client.test()
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
                'clients': [client.make_checkpoint() for client in self.clients]
            },
            os.path.join(self.checkpoint_path, f'model_round{r}.pt')
        )

    def load_checkpoint(self, checkpoint):
        self.backbone.load_state_dict(checkpoint['backbone'])
        for cid, client_checkpoint in enumerate(checkpoint['clients']):
            self.clients[cid].backbone.load_state_dict(checkpoint['backbone'])
            self.clients[cid].load_checkpoint(client_checkpoint)
