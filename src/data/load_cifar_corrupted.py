import os

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Cifar10corrupted(Dataset):
    def __init__(self, root: str, cortype: str, severity: int, transform: transforms):
        if cortype not in ['brightness', 'fog', 'glass_blur', 'motion_blur', 'snow', 'contrast', 'frost',
                           'impulse_noise', 'pixelate', 'spatter', 'defocus_blur', 'gaussian_blur', 'jpeg_compression',
                           'saturate', 'speckle_noise', 'elastic_transform', 'gaussian_noise', 'shot_noise',
                           'zoom_blur']:
            raise AttributeError('corrupt type is not included in CIFAR10-C.')
        self._image_transformer = transform

        images = np.load(os.path.join(root, 'CIFAR-10-C/' + cortype + '.npy'))
        labels = np.load(os.path.join(root, 'CIFAR-10-C/' + 'labels.npy'))

        images = images[severity * 10000: (severity + 1) * 10000]
        labels = labels[severity * 10000: (severity + 1) * 10000]

        self.data = images
        self.targets = labels

        self.num_class = 10

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        img = self._image_transformer(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class Cifar100corrupted(Dataset):
    def __init__(self, root: str, cortype: str, severity: int, transform: transforms):
        if cortype not in ['brightness', 'fog', 'glass_blur', 'motion_blur', 'snow', 'contrast', 'frost',
                           'impulse_noise', 'pixelate', 'spatter', 'defocus_blur', 'gaussian_blur', 'jpeg_compression',
                           'saturate', 'speckle_noise', 'elastic_transform', 'gaussian_noise', 'shot_noise',
                           'zoom_blur']:
            raise AttributeError('corrupt type is not included in CIFAR100-C.')
        self._image_transformer = transform

        images = np.load(os.path.join(root, 'CIFAR-100-C/' + cortype + '.npy'))
        labels = np.load(os.path.join(root, 'CIFAR-100-C/' + 'labels.npy'))

        images = images[severity * 10000: (severity + 1) * 10000]
        labels = labels[severity * 10000: (severity + 1) * 10000]

        self.data = images
        self.targets = labels

        self.num_class = 100

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        img = self._image_transformer(img)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
