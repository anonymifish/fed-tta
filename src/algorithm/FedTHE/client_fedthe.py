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
        self.e_learning_rate = configs.e_learning_rate
        self.alpha = configs.the_alpha
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

        self.e_optimizer = None

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
        self.global_representation = global_representation
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
                        test_history.append(
                            self.alpha * test_representation[i, :] + (1 - self.alpha) * test_history[-1])
                self.history_representation = torch.stack(test_history)

            temperature = torch.hstack(
                (
                    torch.ones((test_representation.shape[0], 1)),
                    torch.ones((test_representation.shape[0], 1)),
                )
            )
            self.e = torch.nn.Parameter(temperature.to(self.device), requires_grad=True)
            self.e.data.fill_(1 / 2)
            self.e_optimizer = torch.optim.Adam([self.e], lr=self.e_learning_rate)

            test_representation = self.backbone.intermediate_forward(data)
            global_out = self.backbone.fc(test_representation)
            personal_out = self.personal_head(test_representation)
            for _ in range(20):
                e_softmax = F.softmax(self.e, dim=1)
                agg_output = e_softmax[:, 0].unsqueeze(1) * global_out.detach() + e_softmax[:, 1].unsqueeze(
                    1) * personal_out.detach()
                test_representation = self.beta * test_representation + (1 - self.beta) * self.history_representation
                p_feature_al = torch.norm((test_representation - self.local_representation.to(self.device)), dim=1)
                g_feature_al = torch.norm((test_representation - self.global_representation.to(self.device)), dim=1)
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                sim = cos(F.softmax(global_out, dim=1).detach(), F.softmax(personal_out, dim=1).detach())
                loss = (
                        -sim * (agg_output.softmax(1) * agg_output.log_softmax(1)).sum(1)
                        + (1 - sim) * (
                                e_softmax[:, 0] * g_feature_al.detach() + e_softmax[:, 1] * p_feature_al.detach())
                ).mean(0)
                self.e_optimizer.zero_grad()
                loss.backward()
                if torch.norm(self.e.grad) < 1e-5:
                    break
                self.e_optimizer.step()

            with torch.no_grad():
                e_softmax = F.softmax(self.e, dim=1)
                test_representation = self.backbone.intermediate_forward(data)
                global_out = self.backbone.fc(test_representation)
                personal_out = self.personal_head(test_representation)
                agg_output = e_softmax[:, 0].unsqueeze(1) * global_out + e_softmax[:, 1].unsqueeze(1) * personal_out
                pred = agg_output.data.max(1)[1]
                accuracy.append(accuracy_score(list(pred.data.cpu().numpy()), list(target.data.cpu().numpy())))

        self.backbone.cpu()
        self.personal_head.cpu()
        return {'acc': sum(accuracy) / len(accuracy)}

    def plain_test(self):
        self.backbone.to(self.device)
        self.personal_head.to(self.device)
        self.backbone.eval()
        self.personal_head.eval()
        accuracy = []
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                z = self.backbone.intermediate_forward(data)
                global_out = self.backbone.fc(z)
                personal_out = self.personal_head(z)
                logits = 0.5 * global_out + 0.5 * personal_out
                pred = logits.data.max(1)[1]
                accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))

        self.backbone.cpu()
        self.personal_head.cpu()
        return {'acc': sum(accuracy) / len(accuracy)}

    def make_checkpoint(self):
        return {
            'fc': self.personal_head.state_dict(),
            'local_representation': self.local_representation,
        }

    def load_checkpoint(self, checkpoint):
        self.personal_head.load_state_dict(checkpoint['fc'])
        self.local_representation = checkpoint['local_representation']
