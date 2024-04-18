import os

import torch
from PIL import Image
from scipy import io
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt

class Digit5Dataset(Dataset):
    def __init__(self, root, dataset_name):
        self.dataset_name = dataset_name
        dataset_path = os.path.join(root, 'Digit-5')
        if dataset_name == 'mnist':
            self.channel = 1
            self.dataset_path = os.path.join(dataset_path, 'mnist_data.mat')
            mnist_data = io.loadmat(self.dataset_path)
            # mnist_train (55000, 28, 28, 1), mnist_label (55000, 10)
            images = mnist_data['train_28']
            label = torch.tensor(mnist_data['label_train'])
            label = torch.argmax(label, dim=1)
        elif dataset_name == 'mnistm':
            self.channel = 3
            self.dataset_path = os.path.join(dataset_path, 'mnistm_with_label.mat')
            mnistm_data = io.loadmat(self.dataset_path)
            # mnistm_train (55000, 28, 28, 3), mnistm_label (55000, 10)
            images = mnistm_data['train']
            label = torch.tensor(mnistm_data['label_train'])
            label = torch.argmax(label, dim=1)
        elif dataset_name == 'svhn':
            self.channel = 3
            self.dataset_path = os.path.join(dataset_path, 'svhn_train_32x32.mat')
            svhn_data = io.loadmat(self.dataset_path)
            # svhn_train (32, 32, 3, 73257), svhn_label (73257, 1)
            images = svhn_data['X'].transpose((3, 0, 1, 2))
            label = torch.tensor(svhn_data['y']).squeeze()
            label[label == 10] = 0
            label = label.long()
        elif dataset_name == 'synthesis':
            self.channel = 3
            self.dataset_path = os.path.join(dataset_path, 'syn_number.mat')
            synthesis_data = io.loadmat(self.dataset_path)
            # synthesis_train (25000, 32, 32, 3), synthesis_label (25000, 1)
            images = synthesis_data['train_data']
            label = torch.tensor(synthesis_data['train_label']).squeeze().long()
        elif dataset_name == 'usps':
            self.channel = 1
            self.dataset_path = os.path.join(dataset_path, 'usps_28x28.mat')
            usps_data = io.loadmat(self.dataset_path)
            # usps_train (7438, 1, 28, 28), usps_label (7438, 1)
            images = usps_data['dataset'][0][0].transpose((0, 2, 3, 1))
            label = torch.tensor(usps_data['dataset'][0][1]).squeeze().long()
        else:
            raise ValueError('Dataset not in digit-5')

        self.data = images
        self.targets = label
        self.num_class = 10

        if self.channel == 1:
            self._image_transformer = transforms.Compose([
                transforms.Resize([28, 28]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self._image_transformer = transforms.Compose([
                transforms.Resize([28, 28]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.channel == 1:
            img = Image.fromarray(img.squeeze(), mode='L')
        else:
            img = Image.fromarray(img, mode='RGB')

        img = self._image_transformer(img)

        return img, target

    def __len__(self):
        return len(self.data)


class PACSDataset(Dataset):
    def __init__(self, root, dataset_name):
        self.dataset_name = dataset_name
        dataset_path = os.path.join(root, 'PACS')
        self.dataset_path = os.path.join(dataset_path, f'{dataset_name}')
        image_folder_dataset = ImageFolder(root=self.dataset_path)
        self.dataset = image_folder_dataset

        self.num_class = 7

        self._image_transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        img, target = self.dataset[index]

        img = self._image_transformer(img)
        target = torch.nn.functional.one_hot(torch.tensor(target), num_classes=self.num_class)

        return img, target

    def __len__(self):
        return len(self.dataset)


class Office10Dataset(Dataset):
    def __init__(self, root, dataset_name):
        self.dataset_name = dataset_name
        dataset_path = os.path.join(root, 'office_caltech_10')
        self.dataset_path = os.path.join(dataset_path, f'{dataset_name}')
        image_folder_dataset = ImageFolder(root=self.dataset_path)
        self.dataset = image_folder_dataset

        self.num_class = 10

        self._image_transformer = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        img, target = self.dataset[index]

        img = self._image_transformer(img)
        target = torch.nn.functional.one_hot(torch.tensor(target), num_classes=self.num_class)

        return img, target

    def __len__(self):
        return len(self.dataset)
