import copy

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from src.algorithm.Base.client_base import BaseClient


class FedTHEClient(BaseClient):
    def __init__(self, cid, device, backbone, configs):
        super(FedTHEClient, self).__init__(cid, device, backbone, configs)
        self.personal_head = torch.nn.Linear(self.backbone.fc.in_features, self.backbone.fc.out_features, False)
        self.personal_head_epoch = configs.personal_head_epoch
        self.alpha = configs.alpha
        self.beta = configs.beta
        self.local_representation = None
        self.global_representation = None

        self.history_representation = None
        self.e = None

        self.optimizer = torch.optim.SGD(
            params=self.backbone.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.personal_head_optimizer = torch.optim.SGD(
            params=self.personal_head.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def train(self):
        global_model = copy.deepcopy(self.backbone)

        self.backbone.to(self.device)
        self.backbone.train()
        accuracy = []

        for epoch in range(self.epochs):
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.backbone(data)

                pred = logits.data.max(1)[1]
                accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))

                self.optimizer.zero_grad()
                loss = F.cross_entropy(logits, target)
                loss.backward()
                self.optimizer.step()

        self.backbone.cpu()

        global_model.to(self.device)
        global_model.eval()
        self.personal_head.to(self.device)
        local_representation_list = []
        record_local_representation_flag = None

        for epoch in range(self.personal_head_epoch):
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                z = global_model.intermediate_forward(data)
                if record_local_representation_flag is None:
                    local_representation_list.append(z)
                logit = self.personal_head(z)
                loss = F.cross_entropy(logit, target)

                self.personal_head_optimizer.zero_grad()
                loss.backward()
                self.personal_head_optimizer.step()
            record_local_representation_flag = torch.concat(local_representation_list).mean(0)
            self.local_representation = record_local_representation_flag.cpu()

        global_model.cpu()
        self.personal_head.cpu()

        return {'backbone': self.backbone.state_dict(), 'accuracy': sum(accuracy) / len(accuracy)}

    def test(self, global_representation):
        self.backbone.to(self.device)
        self.personal_head.to(self.device)
        self.backbone.eval()
        accuracy = []

        for data, target in self.test_dataloader:
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                test_representation = self.backbone.intermediate_forward(data)
                test_history = None
                for i in range(test_representation.shape[0]):
                    if test_history is None and self.history_representation is None:
                        test_history = [test_representation[0, :]]
                    elif test_history is None and self.history_representation is not None:
                        test_history = [self.history_representation[-1, :]]
                    else:
                        test_history.append(self.alpha * test_representation[i, :] + (1 - self.alpha) * test_history[-1])
                self.history_representation = torch.stack(test_history)


    def make_checkpoint(self):
        return {
            'fc': self.personal_head.state_dict(),
            'local_representation': self.local_representation,
        }

    def load_checkpoint(self, checkpoint):
        self.personal_head.load_state_dict(checkpoint['fc'])
        self.local_representation = checkpoint['local_representation']

