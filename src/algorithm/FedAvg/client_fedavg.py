import torch.nn.functional as F
import torch.optim
from sklearn.metrics import accuracy_score

from src.algorithm.Base.client_base import BaseClient


class FedAvgClient(BaseClient):
    def __init__(self, cid, device, backbone, configs):
        super(FedAvgClient, self).__init__(cid, device, backbone, configs)

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
        return {'backbone': self.backbone.state_dict(), 'accuracy': sum(accuracy) / len(accuracy)}

    def test(self):
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
