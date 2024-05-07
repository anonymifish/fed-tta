import copy

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim
from sklearn.metrics import accuracy_score

from src.algorithm.Base.client_base import BaseClient


class MethodClient(BaseClient):
    def __init__(self, cid, device, backbone, configs):
        super(MethodClient, self).__init__(cid, device, backbone, configs)
        self.auxiliary_head = copy.deepcopy(self.backbone.fc)
        self.trade_off = configs.trade_off
        self.add_loss = configs.add_loss
        self.loss_weight = configs.loss_weight

        self.optimizer = torch.optim.SGD(
            params=self.backbone.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.all_optimizer = torch.optim.SGD(
            params=list(self.backbone.parameters()) + list(self.auxiliary_head.parameters()),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.classifier_optimizer = torch.optim.SGD(
            params=self.backbone.fc.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.try_method = configs.try_method

    def train(self):
        self.backbone.to(self.device)
        self.auxiliary_head.to(self.device)
        self.backbone.train()
        self.auxiliary_head.train()
        accuracy = []

        for epoch in range(self.epochs):
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                z = self.backbone.intermediate_forward(data)

                if self.try_method in ['feature_aux-classifier', 'all_training']:
                    optimizer = self.all_optimizer
                else:
                    optimizer = self.optimizer

                if self.try_method in ['feature_aux-classifier', 'feature-classifier']:
                    aux_logits = self.auxiliary_head(z)
                    aux_loss = F.cross_entropy(aux_logits, target)
                    optimizer.zero_grad()
                    aux_loss.backward()
                    optimizer.step()
                elif self.try_method in ['all_training', 'all_training-aux']:
                    aux_logits = self.auxiliary_head(z)
                    original_logits = self.backbone.fc(z)

                    pred = original_logits.data.max(1)[1]
                    accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))
                    aux_loss = F.cross_entropy(aux_logits, target)
                    original_loss = F.cross_entropy(original_logits, target)
                    if self.add_loss:
                        loss = original_loss + self.loss_weight * aux_loss
                    else:
                        loss = self.trade_off * original_loss + (1 - self.trade_off) * aux_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                else:
                    raise ValueError(f'illegal try method {self.try_method}')

        if self.try_method in ['feature_aux-classifier', 'feature-classifier']:
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.backbone(data)
                pred = logits.data.max(1)[1]
                accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))
                loss = F.cross_entropy(logits, target)
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()

        self.backbone.cpu()
        return {
            'backbone': self.backbone.state_dict(),
            'accuracy': sum(accuracy) / len(accuracy),
            'head_states': self.backbone.fc.state_dict(),
        }

    def plain_test(self):
        self.backbone.to(self.device)
        self.backbone.eval()
        accuracy = []
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

        # confidence is not good, then what about logits?
        correct_confidence = []
        incorrect_confidence = []
        correct_entropy = []
        incorrect_entropy = []

        for data, target in self.test_dataloader:
            data, target = data.to(self.device), target.to(self.device)
            logits = self.backbone(data)
            prob = F.softmax(logits, dim=-1)
            entropy = -torch.sum(prob * torch.log(prob), dim=1)
            confidence, pred = torch.max(prob, 1)
            accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))
            for i in range(len(target)):
                if pred[i] == target[i]:
                    correct_entropy.append(entropy[i].item())
                    correct_confidence.append(confidence[i].item())
                else:
                    incorrect_entropy.append(entropy[i].item())
                    incorrect_confidence.append(confidence[i].item())

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidence, bins=20, alpha=0.7, label='Correct Predictions')
        plt.hist(incorrect_confidence, bins=20, alpha=0.7, label='Incorrect Predictions')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(correct_entropy, bins=20, alpha=0.7, label='Correct Predictions')
        plt.hist(incorrect_entropy, bins=20, alpha=0.7, label='Incorrect Predictions')
        plt.xlabel('Entropy')
        plt.ylabel('Count')
        plt.legend()

        plt.savefig('confidence_entropy_distribution.png')

        self.backbone.cpu()
        return {'acc': sum(accuracy) / len(accuracy)}

    def make_checkpoint(self):
        return {'fc': self.backbone.fc.state_dict()}

    def load_checkpoint(self, checkpoint):
        self.backbone.fc.load_state_dict(checkpoint['fc'])
