import copy
import os
import time

import torch
import wandb

from src.algorithm.ATP.client_atp import ATPClient
from src.algorithm.Base.server_base import BaseServer
from utils.logger import logger


class ATPServer(BaseServer):
    def __init__(self, device, backbone, configs):
        super().__init__(device, backbone, configs)
        self.backbone.change_bn()
        self.backbone.eval()

        self.idx_params = [i for i, (name, _) in enumerate(self.backbone.named_parameters()) if 'running' not in name]
        self.idx_stats = [i for i, (name, _) in enumerate(self.backbone.named_parameters()) if 'running' in name]

        self.adapt_lrs = torch.zeros(len(self.backbone.trainable_parameters())).to(self.device)

    def fit(self):
        for r in range(self.global_rounds):
            client_weights = []
            client_accuracy = []
            client_train_time = []
            accum_adapt_lrs = torch.zeros_like(self.adapt_lrs)

            active_clients = self.select_clients()
            logger.info(f'round{r}')
            for client in active_clients:
                client: ATPClient
                client.backbone.load_state_dict(self.backbone.state_dict())
                client.adapt_lrs = copy.deepcopy(self.adapt_lrs)
                client_weights.append(len(client.train_dataloader))
                start_time = time.time()
                report = client.train()
                end_time = time.time()
                client_accuracy.append(report['accuracy'])
                accum_adapt_lrs += report['adapt_lrs']
                client_train_time.append(end_time - start_time)
                logger.info(f"client{client.cid} training time: {end_time - start_time}")

            self.adapt_lrs = accum_adapt_lrs / len(active_clients)

            logger.info(f'average client train time: {sum(client_train_time) / len(client_train_time)}')
            logger.info(
                f'average client accuracy: {sum([client_accuracy[i] * client_weights[i] / sum(client_weights) for i in range(len(active_clients))])}')
            if not self.debug:
                wandb.log({
                    'accuracy': sum([client_accuracy[i] * client_weights[i] / sum(client_weights) for i in
                                     range(len(active_clients))]),
                    'train_time': sum(client_train_time) / len(client_train_time),
                })

            if (r + 1) % 5 == 0:
                self.make_checkpoint(r + 1)

    def plain_test(self):
        accuracy = []
        client_weights = []
        for client in self.clients:
            client.adapt_lrs = copy.deepcopy(self.adapt_lrs)
            client_weights.append(len(client.test_dataloader))
            report = client.test()
            accuracy.append(report['acc'])

        return sum([accuracy[i] * client_weights[i] / sum(client_weights) for i in range(len(self.clients))])

    def test(self):
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
                'adapt_lrs': self.adapt_lrs,
            },
            os.path.join(self.checkpoint_path, f'atp_model_round{r}.pt')
        )

    def load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, dict) and 'backbone' in checkpoint:
            self.backbone.load_state_dict(checkpoint['backbone'])
            self.adapt_lrs = checkpoint['adapt_lrs']
            for client in self.clients:
                client.backbone.load_state_dict(checkpoint['backbone'])
        else:
            self.backbone.load_state_dict(checkpoint)
            for client in self.clients:
                client.backbone.load_state_dict(checkpoint)
