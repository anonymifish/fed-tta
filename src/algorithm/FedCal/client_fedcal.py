import copy

import torch.nn.functional as F
import torch.optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src.algorithm.Base.client_base import BaseClient


def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
    Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


class FedCalClient(BaseClient):
    def __init__(self, cid, device, backbone, configs):
        super(FedCalClient, self).__init__(cid, device, backbone, configs)
        self.num_class = configs.num_class
        self.sample_per_class = torch.zeros(self.num_class)

        self.optimizer = torch.optim.SGD(
            params=self.backbone.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def train(self):
        self.backbone.to(self.device)
        self.backbone.train()
        accuracy = []

        temp_head_stat = copy.deepcopy(self.backbone.fc.state_dict())
        for epoch in range(self.epochs):
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.backbone(data)

                pred = logits.data.max(1)[1]
                accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))
                loss = balanced_softmax_loss(target, logits, self.sample_per_class)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.backbone.fc.load_state_dict(temp_head_stat)

        self.backbone.cpu()
        return {'backbone': self.backbone.state_dict(), 'accuracy': sum(accuracy) / len(accuracy)}

    def test(self):
        self.backbone.to(self.device)
        self.backbone.eval()
        accuracy = []

        p_s = torch.ones(self.num_class)
        p_s = p_s / self.num_class
        p_t = copy.deepcopy(p_s)

        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.backbone(data)
                prob = F.softmax(logits)
                for iter in range(10):
                    p_con = (prob * p_t / p_s) / (prob * p_t / p_s).sum()
                    p_t = p_con.sum().mean()
                prob = prob * p_t / p_s
                pred = prob.data.max(1)[1]
                accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))
        self.backbone.cpu()
        return {'acc': sum(accuracy) / len(accuracy)}

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

    def set_train_set(self, train_set):
        self.train_set = train_set
        self.train_dataloader = DataLoader(
            dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=3, pin_memory=False,
        )

        for _, targets in self.train_dataloader:
            for sample_label in targets:
                self.sample_per_class[sample_label.item()] += 1
