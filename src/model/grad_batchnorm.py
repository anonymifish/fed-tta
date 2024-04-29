import torch
from torch import nn


class GradBatchNorm1d(nn.Module):
    def __init__(self, bn):
        super(GradBatchNorm1d, self).__init__()

        self.num_features = bn.num_features
        self.eps = bn.eps
        self.num_batches_tracked = bn.num_batches_tracked
        self.running_mean = nn.parameter.Parameter(bn.running_mean.detach().clone(), True)
        self.running_var = nn.parameter.Parameter(bn.running_var.detach().clone(), True)
        self.weight = nn.parameter.Parameter(bn.weight.detach().clone(), True)
        self.bias = nn.parameter.Parameter(bn.bias.detach().clone(), True)

        self.snapshot_mean = None
        self.snapshot_var = None

        self.training = True

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False

    def forward(self, x):
        with torch.no_grad():
            x_t = x.data.permute((1, 0)).reshape((self.num_features, -1)).detach().clone()
            self.snapshot_mean = x_t.mean(dim=1)
            self.snapshot_var = x_t.var(dim=1)
            self.num_batches_tracked += 1

        if not self.training:
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        else:
            var_forward = x_t.var(dim=1, unbiased=False)
            x = (x - self.snapshot_mean) / torch.sqrt(var_forward + self.eps)

        x = x * self.weight + self.bias

        return x

    def set_running_stat_grad(self):
        with torch.no_grad():
            self.running_mean.grad = self.running_mean.data - self.snapshot_mean
            self.running_var.grad = self.running_var.data - self.snapshot_var

    def clip_running_var(self):
        with torch.no_grad():
            self.running_var.clamp_(min=0)


class GradBatchNorm2d(nn.Module):
    def __init__(self, bn):
        super(GradBatchNorm2d, self).__init__()

        self.num_features = bn.num_features
        self.eps = bn.eps
        self.num_batches_tracked = bn.num_batches_tracked
        self.running_mean = nn.parameter.Parameter(bn.running_mean.detach().clone(), True)
        self.running_var = nn.parameter.Parameter(bn.running_var.detach().clone(), True)
        self.weight = nn.parameter.Parameter(bn.weight.detach().clone(), True)
        self.bias = nn.parameter.Parameter(bn.bias.detach().clone(), True)

        self.snapshot_mean = None
        self.snapshot_var = None

        self.training = True

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.training = False

    def forward(self, x):
        with torch.no_grad():
            x_t = x.data.permute((1, 0, 2, 3)).reshape((self.num_features, -1)).detach().clone()
            self.snapshot_mean = x_t.mean(dim=1)
            self.snapshot_var = x_t.var(dim=1)
            self.num_batches_tracked += 1

        if not self.training:
            x = (x - self.running_mean.view((-1, 1, 1))) / torch.sqrt(self.running_var.view((-1, 1, 1)) + self.eps)
        else:
            var_forward = x_t.var(dim=1, unbiased=False)
            x = (x - self.snapshot_mean.view((-1, 1, 1))) / torch.sqrt(var_forward.view((-1, 1, 1)) + self.eps)

        x = x * self.weight.view((-1, 1, 1)) + self.bias.view((-1, 1, 1))

        return x

    def set_running_stat_grad(self):
        with torch.no_grad():
            self.running_mean.grad = self.running_mean.data - self.snapshot_mean
            self.running_var.grad = self.running_var.data - self.snapshot_var

    def clip_running_var(self):
        with torch.no_grad():
            self.running_var.clamp_(min=0)
