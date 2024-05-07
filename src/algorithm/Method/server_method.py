import copy
import os
import time

import torch
import wandb

from src.algorithm.Base.server_base import BaseServer
from src.algorithm.Method.client_method import MethodClient
from utils.logger import logger


class MethodServer(BaseServer):
    def __init__(self, device, backbone, configs, profiler):
        super().__init__(device, backbone, configs, profiler)
        self.avg_head = configs.avg_head
        self.aux_ratio = configs.aux_ratio
        head_state_dict = self.backbone.fc.state_dict()
        self.aux_heads = {client.cid: copy.deepcopy(head_state_dict) for client in self.clients}

    def auxiliary_head_select(self):
        state_vect = []
        for head_state in self.aux_heads.values():
            state_vect.append(torch.cat([param.view(-1) for param in head_state.values()]))
        state_vect_matrix = torch.stack(state_vect)

        distance = torch.norm(state_vect_matrix.unsqueeze(1) - state_vect_matrix.unsqueeze(0), dim=2)
        max_distance, max_indices = torch.topk(distance, int(len(self.clients) * self.aux_ratio), dim=1)

        ret = []
        for cid, _ in enumerate(self.aux_heads):
            id_list = max_indices[cid].int().tolist()
            logger.info(f'for client{cid}, client{id_list}\'s head has the max distance {max_distance[cid]}')
            aux_head_state = self.model_average([self.aux_heads[i] for i in id_list], [1 for _ in id_list])
            ret.append(aux_head_state)
        return ret

    def fit(self):
        for r in range(self.global_rounds):
            client_weights = []
            client_net_states = []
            client_accuracy = []
            client_train_time = []

            active_clients = self.select_clients()
            for client in active_clients:
                client: MethodClient
                client_weights.append(len(client.train_dataloader))
                start_time = time.time()
                report = client.train()
                end_time = time.time()
                client_net_states.append(report['backbone'])
                client_accuracy.append(report['accuracy'])
                self.aux_heads[client.cid] = copy.deepcopy(report['head_states'])
                client_train_time.append(end_time - start_time)
                logger.info(f"client{client.cid} training time: {end_time - start_time}")

            global_net_state = self.model_average(client_net_states, client_weights)
            aux_heads = self.auxiliary_head_select()
            for client in self.clients:
                if self.avg_head:
                    client.backbone.load_state_dict(global_net_state)
                else:
                    tmp_fc_state = copy.deepcopy(client.backbone.fc.state_dict())
                    client.backbone.load_state_dict(global_net_state)
                    client.backbone.fc.load_state_dict(tmp_fc_state)
                client.auxiliary_head.load_state_dict(aux_heads[client.cid])
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
                'clients': [client.make_checkpoint() for client in self.clients],
            },
            os.path.join(self.checkpoint_path, f'model_round{r}.pt')
        )

    def load_checkpoint(self, checkpoint):
        self.backbone.load_state_dict(checkpoint['backbone'])
        for cid, client_checkpoint in enumerate(checkpoint['clients']):
            self.clients[cid].backbone.load_state_dict(checkpoint['backbone'])
            self.clients[cid].load_checkpoint(client_checkpoint)
