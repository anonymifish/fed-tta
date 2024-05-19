import copy
import os

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim
from sklearn.metrics import accuracy_score

from src.algorithm.Base.client_base import BaseClient
from src.algorithm.T3A import T3A


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

        self.checkpoint_path = configs.checkpoint_path

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

        adapt_model = MethodAdapt(self.backbone, 20, 25)
        t3a_adapt_model = T3A(self.backbone, -1)

        for data, target in self.test_dataloader:
            data, target = data.to(self.device), target.to(self.device)
            # logits = adapt_model.forward(data)
            logits = t3a_adapt_model(data)
            # logits = self.backbone(data)
            pred = logits.data.max(1)[1]
            accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))
            # adapt_model.statistic_update(logits, target)

        # adapt_model.save_statistic(self.checkpoint_path, self.cid)
        self.backbone.cpu()
        return {'acc': sum(accuracy) / len(accuracy)}

    def make_checkpoint(self):
        return {'fc': self.backbone.fc.state_dict()}

    def load_checkpoint(self, checkpoint):
        self.backbone.fc.load_state_dict(checkpoint['fc'])


class MethodAdapt:
    def __init__(self, backbone, filter_number, threshold):
        self.backbone = backbone
        self.num_class = self.backbone.fc.out_features
        self.filter_number = filter_number
        self.threshold = threshold
        self.classifier_weight = copy.deepcopy(self.backbone.fc.weight.data)

        self.support_feature = [self.classifier_weight[i].unsqueeze(0) for i in range(self.num_class)]
        self.logit = [(self.classifier_weight @ self.classifier_weight[i]).max() for i in range(self.num_class)]

        self.optimizer = torch.optim.SGD(params=self.backbone.parameters(), lr=0.01, momentum=0, weight_decay=5e-4)

        self.correct_confidence = {label: [] for label in range(self.num_class)}
        self.incorrect_confidence = {label: [] for label in range(self.num_class)}
        self.correct_entropy = {label: [] for label in range(self.num_class)}
        self.incorrect_entropy = {label: [] for label in range(self.num_class)}
        self.correct_logit = {label: [] for label in range(self.num_class)}
        self.incorrect_logit = {label: [] for label in range(self.num_class)}

    def forward(self, data):
        feature = self.backbone.intermediate_forward(data)
        logits = self.backbone.fc(feature)
        # max_logit, pred = torch.max(logits, 1)
        #
        # adapt_indices = torch.nonzero(max_logit >= self.threshold).squeeze()
        # if adapt_indices.numel() != 0:
        #     adapt_feature = feature[adapt_indices]
        #     std_feature = self.get_support(pred[adapt_indices])
        #
        #     loss = torch.norm(adapt_feature - std_feature, p=2)
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()
        #
        #     feature = self.backbone.intermediate_forward(data)
        #     logits = self.backbone.fc(feature)
        #
        # self.update_support(feature, logits)
        return logits

    def get_support(self, labels):
        return torch.stack([self.support_feature[i].mean(dim=0) for i in labels.unsqueeze(0)])

    def update_support(self, new_feature, new_logits):
        max_logit, pred = torch.max(new_logits, 1)
        update_indices = torch.nonzero(max_logit >= self.threshold).squeeze()

        if update_indices.numel() != 0:
            update_max_logit = max_logit[update_indices]
            update_feature = new_feature[update_indices]
            update_pred = pred[update_indices]
            for i in range(self.num_class):
                if update_max_logit[update_pred == i].numel() == 0:
                    continue
                self.logit[i] = torch.cat([self.logit[i], update_max_logit[update_pred == i]])
                self.support_feature[i] = torch.cat([self.support_feature[i], update_feature[update_pred == i]])
                if self.logit[i].size(0) > self.filter_number:
                    _, top_indices = torch.topk(self.logit[i], self.filter_number)
                    self.logit[i] = self.logit[i][top_indices]
                    self.support_feature[i] = self.support_feature[i][top_indices]

    def statistic_update(self, logits, target):
        max_logit, pred = torch.max(logits, 1)
        prob = F.softmax(logits, dim=-1)
        entropy = -torch.sum(prob * torch.log(prob), dim=1)
        confidence, _ = torch.max(prob, 1)
        for i in range(len(target)):
            if pred[i] == target[i]:
                self.correct_logit[pred[i].item()].append(max_logit[i].item())
                self.correct_entropy[pred[i].item()].append(entropy[i].item())
                self.correct_confidence[pred[i].item()].append(confidence[i].item())
            else:
                self.incorrect_logit[pred[i].item()].append(max_logit[i].item())
                self.incorrect_entropy[pred[i].item()].append(entropy[i].item())
                self.incorrect_confidence[pred[i].item()].append(confidence[i].item())

    def save_statistic(self, checkpoint_path, cid):
        figure_save_path = os.path.join(checkpoint_path, f"client{cid}")
        os.makedirs(figure_save_path, exist_ok=True)
        for label in range(self.num_class):
            plt.figure(figsize=(40, 15))
            for i, name in enumerate(['confidence', 'entropy', 'logit']):
                plt.subplot(1, 3, i + 1)
                plt.hist(eval(f'self.correct_{name}')[label], bins=20, alpha=0.7, label='Correct Predictions')
                plt.hist(eval(f'self.incorrect_{name}')[label], bins=20, alpha=0.7, label='Incorrect Predictions')
                plt.xlabel(f'{name}', fontsize=50)
                plt.ylabel('count', fontsize=50)
                plt.legend(fontsize=40)
                plt.xticks(fontsize=40)
                plt.yticks(fontsize=40)
            plt.savefig(os.path.join(figure_save_path, f'class{label}.png'))

        # correct_all_confidence = []
        # correct_all_entropy = []
        # correct_all_logit = []
        # incorrect_all_confidence = []
        # incorrect_all_entropy = []
        # incorrect_all_logit = []
        # for i, name in enumerate(['confidence', 'entropy', 'logit']):
        #     for label in range(self.num_class):
        #         eval(f'correct_all_{name}').extend(eval(f'self.correct_{name}')[label])
        #         eval(f'incorrect_all_{name}').extend(eval(f'self.incorrect_{name}')[label])
        #
        # figure_save_path = os.path.join(checkpoint_path, f"client{cid}_all")
        # os.makedirs(figure_save_path, exist_ok=True)
        # plt.figure(figsize=(18, 6))
        # for i, name in enumerate(['confidence', 'entropy', 'logit']):
        #     plt.subplot(1, 3, i + 1)
        #     plt.hist(eval(f'correct_all_{name}'), bins=20, alpha=0.7, label='Correct Predictions')
        #     plt.hist(eval(f'incorrect_all_{name}'), bins=20, alpha=0.7, label='Incorrect Predictions')
        #     plt.xlabel(f'{name}')
        #     plt.ylabel('count')
        #     plt.legend()
        # plt.savefig(os.path.join(figure_save_path, f'client_{cid}.png'))