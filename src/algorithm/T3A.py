import copy

import torch


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def get_max_logits(x: torch.Tensor) -> torch.Tensor:
    return x.max(dim=1).values

class T3A(torch.nn.Module):
    def __init__(self, backbone, filter_number, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = copy.deepcopy(backbone)
        self.classifier = self.backbone.fc

        self.filter_number = filter_number  # 1, 5, 20, 50, 100, -1
        self.num_class = self.classifier.out_features
        self.softmax = torch.nn.Softmax(-1)

        self.warmup_supports = self.classifier.weight.data
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=self.num_class).float()

        self.warmup_max_logits = get_max_logits(warmup_prob)

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.max_logits = self.warmup_max_logits.data

    def forward(self, x):
        z = self.backbone.intermediate_forward(x).detach()
        p = self.classifier(z).detach()
        yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_class).float()
        ent = softmax_entropy(p)

        max_logits = get_max_logits(p)

        indices = max_logits > 25

        self.supports = self.supports.to(z.device)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)
        self.supports = torch.cat([self.supports, z[indices]])
        self.labels = torch.cat([self.labels, yhat[indices]])
        self.ent = torch.cat([self.ent, ent[indices]])

        self.max_logits = self.max_logits.to(z.device)
        self.max_logits = torch.cat([self.max_logits, max_logits[indices]])

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ labels)

        for i in range(self.num_class):
            weights[:, i] /= labels[:, i].sum()
        return z @ weights
        # return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_number = self.filter_number

        max_logits_s = self.max_logits

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).to(ent_s.device)
        for i in range(self.num_class):
            # _, indices2 = torch.sort(ent_s[y_hat == i])
            _, indices2 = torch.sort(max_logits_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_number])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        self.max_logits = self.max_logits[indices]

        return self.supports, self.labels

    def predict(self, x):
        return self(x)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.max_logits = self.warmup_max_logits.data
