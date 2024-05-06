import torch
import torch.nn.functional as F

from src.model.grad_batchnorm import GradBatchNorm2d, GradBatchNorm1d


class SimpleCNN28(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN28, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 64)
        self.fc = torch.nn.Linear(64, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(-1, 64 * 5 * 5)
        out = F.relu(self.fc1(out))
        out = self.fc(out)
        return out

    def intermediate(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(-1, 64 * 5 * 5)
        out = F.relu(self.fc1(out))
        return out

    def change_bn(self):
        pass

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_running_stat_grads(self):
        pass

    def clip_bn_running_vars(self):
        pass


class SimpleCNN(torch.nn.Module):
    def __init__(self, n_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 64)
        self.fc = torch.nn.Linear(64, n_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(-1, 64 * 5 * 5)
        out = F.relu(self.fc1(out))
        out = self.fc(out)
        return out

    def intermediate(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(-1, 64 * 5 * 5)
        out = F.relu(self.fc1(out))
        return out

    def change_bn(self):
        pass

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_running_stat_grads(self):
        pass

    def clip_bn_running_vars(self):
        pass


class ShallowCNN(torch.nn.Module):

    def __init__(self, shape_in, n_classes=10):
        super(ShallowCNN, self).__init__()

        in_channels = shape_in[0]
        h = ((((shape_in[1] - 2) // 2) - 2) // 2) - 2
        w = ((((shape_in[2] - 2) // 2) - 2) // 2) - 2

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), bias=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=64)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), bias=False)
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(in_features=h * w * 64, out_features=64)
        self.bn4 = torch.nn.BatchNorm1d(num_features=64)
        self.relu4 = torch.nn.ReLU()
        self.fc = torch.nn.Linear(in_features=64, out_features=n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = x.view(x.shape[0], -1)
        x = self.linear4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.fc(x)

        return x

    def intermediate_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = x.view(x.shape[0], -1)
        x = self.linear4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        return x

    def change_bn(self):
        self.bn1 = GradBatchNorm2d(self.bn1)
        self.bn2 = GradBatchNorm2d(self.bn2)
        self.bn3 = GradBatchNorm2d(self.bn3)
        self.bn4 = GradBatchNorm1d(self.bn4)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_running_stat_grads(self):
        for m in self.modules():
            if isinstance(m, GradBatchNorm2d) or isinstance(m, GradBatchNorm1d):
                m.set_running_stat_grad()

    def clip_bn_running_vars(self):
        for m in self.modules():
            if isinstance(m, GradBatchNorm2d) or isinstance(m, GradBatchNorm1d):
                m.clip_bn_running_var()


def cnn(description, shape, n_classes):
    if description == 'simplecnn':
        if shape[1] == 28:
            return SimpleCNN28()
        else:
            return SimpleCNN(n_classes)
    elif description == 'shallowcnn':
        return ShallowCNN(shape, n_classes)
    else:
        raise NotImplementedError('CNN\'s should be SimpleCNN or ShallowCNN')
