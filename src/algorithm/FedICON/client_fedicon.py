import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.algorithm.Base.client_base import BaseClient
from src.data.augpair_dataset import AugPairDataset


class FedICONClient(BaseClient):
    def __init__(self, cid, device, backbone, configs):
        super(FedICONClient, self).__init__(cid, device, backbone, configs)
        self.temperature = configs.temperature
        self.unsupervised_temperature = configs.unsupervised_temperature
        self.finetune_epochs = configs.finetune_epochs
        self.icon_learning_rate = configs.icon_learning_rate

        self.original_set = None
        self.original_dataloader = None
        self.finetune_optimizer = torch.optim.SGD(
            params=self.backbone.fc.parameters(),
            lr=self.learning_rate,
        )

        self.avg_optimizer = torch.optim.SGD(
            params=self.backbone.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        self.icon_optimizer = torch.optim.SGD(
            params=self.backbone.parameters(),
            lr=self.icon_learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def fedavg_train(self):
        self.backbone.to(self.device)
        self.backbone.train()
        accuracy = []

        for epoch in range(self.epochs):
            for data, target in self.original_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.backbone(data)

                pred = logits.data.max(1)[1]
                accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))

                self.avg_optimizer.zero_grad()
                loss = F.cross_entropy(logits, target)
                loss.backward()
                self.avg_optimizer.step()

        self.backbone.cpu()
        return {'backbone': self.backbone.state_dict(), 'accuracy': sum(accuracy) / len(accuracy)}

    def train(self):
        self.backbone.to(self.device)
        self.backbone.train()

        for epoch in range(self.epochs):
            for x1, x2, y in self.train_dataloader:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                z1 = self.backbone.intermediate_forward(x1)
                z2 = self.backbone.intermediate_forward(x2)

                z_all = torch.concat((z1, z2), dim=0)
                labels_all = torch.concat((y, y), dim=0)

                sim_matrix = z_all @ z_all.t() / self.temperature
                sim_score = torch.exp(sim_matrix)

                diag_mask = 1 - torch.eye(labels_all.size(0), device=self.device)
                sim_score = sim_score * diag_mask

                labels_equal = labels_all.unsqueeze(1) == labels_all.unsqueeze(0)
                numerator = sim_score * labels_equal.float()
                denominator = sim_score.sum(dim=1, keepdim=True)

                positive = numerator.sum(dim=1)
                loss = -torch.log(positive / denominator).mean()

                self.icon_optimizer.zero_grad()
                loss.backward()
                self.icon_optimizer.step()

        self.backbone.cpu()
        return {'backbone': self.backbone.state_dict()}

    def finetune(self):
        self.backbone.to(self.device)
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

        for epoch in range(self.finetune_epochs):
            for data, target in self.original_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.backbone(data)

                self.finetune_optimizer.zero_grad()
                loss = F.cross_entropy(logits, target)
                loss.backward()
                self.finetune_optimizer.step()

        for param in self.backbone.parameters():
            param.requires_grad = True

        self.backbone.cpu()

        return {'fc': self.backbone.fc.state_dict()}

    def set_train_set(self, train_set):
        self.original_set = train_set
        self.original_dataloader = DataLoader(
            dataset=self.original_set, batch_size=self.batch_size, shuffle=True, num_workers=3, pin_memory=False,
        )

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = std = [x / 255 for x in [63.0, 62.1, 66.7]]

        class DeNormalize(object):
            def __init__(self):
                self.mean = mean
                self.std = std

            def __call__(self, tensor):
                for t, m, s in zip(tensor, self.mean, self.std):
                    t.mul_(s).add_(m)
                return tensor

        denormalize = DeNormalize()
        transform = torchvision.transforms.Compose([
            denormalize,
            # TODO
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0), antialias=True),
            transforms.RandomApply([torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std)
        ])
        self.train_set = AugPairDataset(train_set, transform=transform)

        self.train_dataloader = DataLoader(
            dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=3, pin_memory=False,
        )

    def set_test_set(self, test_set, test_batch_size):
        self.test_batch_size = test_batch_size

        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = std = [x / 255 for x in [63.0, 62.1, 66.7]]

        class DeNormalize(object):
            def __init__(self):
                self.mean = mean
                self.std = std

            def __call__(self, tensor):
                for t, m, s in zip(tensor, self.mean, self.std):
                    t.mul_(s).add_(m)
                return tensor

        denormalize = DeNormalize()
        transform = torchvision.transforms.Compose([
            denormalize,
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0), antialias=True),
            transforms.RandomApply([torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean, std)
        ])
        self.test_set = AugPairDataset(test_set, transform=transform)

        self.test_dataloader = DataLoader(
            dataset=self.test_set, batch_size=self.test_batch_size, shuffle=True, num_workers=3, pin_memory=False,
        )

    @staticmethod
    def unsupervised_contrastive_loss(x1, x2, t=0.1):
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        batch_size = x1.size(0)
        out = torch.cat([x1, x2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / t)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / t)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss

    def test(self):
        self.backbone.to(self.device)
        accuracy = []

        for x1, x2, y in self.test_dataloader:
            if x1.shape[0] == 1:
                continue
            self.backbone.train()
            x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
            z1 = self.backbone.intermediate_forward(x1)
            z2 = self.backbone.intermediate_forward(x2)
            z_all = torch.concat((z1, z2), dim=0)
            loss = self.unsupervised_contrastive_loss(z1, z1, t=self.unsupervised_temperature)
            self.icon_optimizer.zero_grad()
            loss.backward()
            self.icon_optimizer.step()
            self.backbone.eval()
            with torch.no_grad():
                logit = self.backbone(x1)
                pred = logit.data.max(1)[1]
                accuracy.append(accuracy_score(list(pred.data.cpu().numpy()), list(y.data.cpu().numpy())))

        self.backbone.cpu()
        return {'acc': sum(accuracy) / len(accuracy)}

    def plain_test(self):
        self.backbone.to(self.device)
        self.backbone.eval()
        accuracy = []
        with torch.no_grad():
            for data, _, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                logits = self.backbone(data)
                pred = logits.data.max(1)[1]
                accuracy.append(accuracy_score(list(target.data.cpu().numpy()), list(pred.data.cpu().numpy())))

        self.backbone.cpu()
        return {'acc': sum(accuracy) / len(accuracy)}

    def make_checkpoint(self):
        return {'fc': self.backbone.fc.state_dict()}

    def load_checkpoint(self, checkpoint):
        self.backbone.fc.load_state_dict(checkpoint['fc'])
