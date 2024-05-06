import copy

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from src.algorithm.Base.client_base import BaseClient


def entropy_loss(x, y):
    return -(x.softmax(dim=1) * x.log_softmax(dim=1)).sum(dim=1).mean(dim=0)


class ATPClient(BaseClient):
    def __init__(self, cid, device, backbone, confgis):
        super(ATPClient, self).__init__(cid, device, backbone, confgis)
        self.backbone.change_bn()
        self.backbone.eval()

        self.adapt_lrs = None

    def train(self):
        self.backbone.to(self.device)
        accuracy = []

        state = copy.deepcopy(self.backbone.state_dict())
        for data, target in self.train_dataloader:
            self.backbone.load_state_dict(state)
            data, target = data.to(self.device), target.to(self.device)

            logits = self.backbone(data)
            loss = entropy_loss(logits, target)
            self.backbone.zero_grad()
            loss.backward()
            self.backbone.set_running_stat_grads()
            unsupervised_grad = [p.grad.clone() for p in self.backbone.trainable_parameters()]
            with torch.no_grad():
                for i, (param, grad) in enumerate(zip(self.backbone.trainable_parameters(), unsupervised_grad)):
                    param -= self.adapt_lrs[i] * grad
            self.backbone.zero_grad()
            self.backbone.clip_bn_running_vars()

            logits = self.backbone(data)
            loss = F.cross_entropy(logits, target)
            supervised_grad = torch.autograd.grad(loss, self.backbone.trainable_parameters())
            with torch.no_grad():
                g = torch.zeros_like(self.adapt_lrs)
                l = torch.zeros_like(self.adapt_lrs)
                for i, (grad1, grad2) in enumerate(zip(supervised_grad, unsupervised_grad)):
                    g[i] += (grad1 * grad2).sum()
                    l[i] += grad1.numel()
                g /= torch.sqrt(l)
                self.adapt_lrs += self.learning_rate * g

            with torch.no_grad():
                pred = logits.data.max(1)[1]
                accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))

        self.backbone.cpu()
        return {'adapt_lrs': self.adapt_lrs, 'accuracy': sum(accuracy) / len(accuracy)}

    def plain_test(self):
        self.backbone.to(self.device)
        self.backbone.eval()
        accuracy = []
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.backbone(data)
                pred = logits.data.max(1)[1]
                accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))

        self.backbone.cpu()
        return {'acc': sum(accuracy) / len(accuracy)}

    def test(self):
        self.backbone.to(self.device)
        self.backbone.eval()
        accuracy = []

        state = copy.deepcopy(self.backbone.state_dict())
        for data, target in self.test_dataloader:
            self.backbone.load_state_dict(state)
            data, target = data.to(self.device), target.to(self.device)

            logits = self.backbone(data)
            loss = entropy_loss(logits, target)
            self.backbone.zero_grad()
            loss.backward()
            self.backbone.set_running_stat_grads()
            unsupervised_grad = [p.grad.clone() for p in self.backbone.trainable_parameters()]
            with torch.no_grad():
                for i, (param, grad) in enumerate(zip(self.backbone.trainable_parameters(), unsupervised_grad)):
                    param -= self.adapt_lrs[i] * grad
            self.backbone.zero_grad()
            self.backbone.clip_bn_running_vars()

            with torch.no_grad():
                logits = self.backbone(data)
                pred = logits.data.max(1)[1]
                accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))

        self.backbone.cpu()
        return {'acc': sum(accuracy) / len(accuracy)}
