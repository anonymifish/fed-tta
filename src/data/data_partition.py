import logging

import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms

from src.data.load_cifar_corrupted import Cifar10corrupted, Cifar100corrupted
from src.data.load_domain_datasets import Digit5Dataset, PACSDataset, Office10Dataset


def create_dirichlet_distribution(alpha: float, num_client: int, num_class: int, seed: int):
    random_number_generator = np.random.default_rng(seed)
    distribution = random_number_generator.dirichlet(np.repeat(alpha, num_client), size=num_class).transpose()
    distribution /= distribution.sum()
    return distribution


def split_by_distribution(targets, distribution):
    num_client, num_class = distribution.shape[0], distribution.shape[1]
    sample_number = np.floor(distribution * len(targets))
    class_idx = {class_label: np.where(targets == class_label)[0] for class_label in range(num_class)}

    idx_start = np.zeros((num_client + 1, num_class), dtype=np.int32)
    for i in range(0, num_client):
        idx_start[i + 1] = idx_start[i] + sample_number[i]

    client_samples = {idx: {} for idx in range(num_client)}
    for client_idx in range(num_client):
        samples_idx = np.array([], dtype=np.int32)
        for class_label in range(num_class):
            start, end = idx_start[client_idx, class_label], idx_start[client_idx + 1, class_label]
            samples_idx = (np.concatenate((samples_idx, class_idx[class_label][start:end].tolist())).astype(np.int32))
        client_samples[client_idx] = samples_idx

    return client_samples


def load_cifar(configs, corrupt_list=None):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if configs.step == 'train':
        if configs.dataset == 'cifar10':
            train_data = CIFAR10(root=configs.dataset_path, download=True, train=True, transform=trans)
            num_class = 10
        else:
            train_data = CIFAR100(root=configs.dataset_path, download=True, train=True, transform=trans)
            num_class = 100

        distribution = create_dirichlet_distribution(configs.alpha, configs.num_client, configs.num_class,
                                                     configs.dataset_seed)
        train_split = split_by_distribution(np.array(train_data.targets), distribution)
        train_datasets = [Subset(train_data, train_split[idx]) for idx in range(configs.num_client)]

        logging.info(f'------ dirichlet distribution with alpha = {configs.alpha}, {configs.num_client} clients ------')
        logging.info(f'train datasets: {[len(dataset) for dataset in train_datasets]}')
        return train_datasets, num_class
    else:
        if configs.dataset == 'cifar10':
            test_data = CIFAR10(root=configs.dataset_path, download=True, train=False, transform=trans)
            cor_test = dict()
            for cor_type in corrupt_list:
                cor_test[cor_type] = [Cifar10corrupted(
                    root=configs.dataset_path, cortype=cor_type, severity=i, transform=trans) for i in range(5)]
            num_class = 10
        else:
            test_data = CIFAR100(root=configs.dataset_path, download=True, train=False, transform=trans)
            cor_test = dict()
            for cor_type in corrupt_list:
                cor_test[cor_type] = [Cifar100corrupted(
                    root=configs.dataset_path, cortype=cor_type, severity=i, transform=trans) for i in range(5)]
            num_class = 100

        distribution = create_dirichlet_distribution(configs.alpha, configs.num_client, configs.num_class,
                                                     configs.dataset_seed)
        no_shift_split = split_by_distribution(np.array(test_data.targets), distribution)
        covariate_shift_split = [split_by_distribution(np.array(cor_test[corrupt_list[0]][i].targets), distribution) for
                                 i in range(5)]

        new_distribution = create_dirichlet_distribution(configs.alpha, configs.num_client, configs.num_class,
                                                         configs.new_dataset_seed)
        label_shift_split = split_by_distribution(np.array(test_data.targets), new_distribution)
        hybrid_shift_split = [split_by_distribution(np.array(cor_test[corrupt_list[0]][i].targets), new_distribution)
                              for i in range(5)]

        no_shift = [Subset(test_data, no_shift_split[idx]) for idx in range(configs.num_client)]
        label_shift = [Subset(test_data, label_shift_split[idx]) for idx in range(configs.num_client)]
        covariate_shift, hybrid_shift = dict(), dict()
        for idx, cor_type in enumerate(corrupt_list):
            covariate_shift[cor_type] = []
            hybrid_shift[cor_type] = []
            for severity in range(5):
                covariate_shift[cor_type].append(
                    [Subset(cor_test[cor_type][severity], covariate_shift_split[severity][idx]) for idx in
                     range(configs.num_client)])
                hybrid_shift[cor_type].append(
                    [Subset(cor_test[cor_type][severity], hybrid_shift_split[severity][idx]) for idx in
                     range(configs.num_client)])

        return no_shift, label_shift, covariate_shift, hybrid_shift, num_class


def load_domains(configs):
    domain_datasets = dict()
    if configs.dataset == 'digit-5':
        dataset_names = ['mnist', 'mnistm', 'svhn', 'synthesis', 'usps']
        for name in dataset_names:
            domain_datasets[name] = Digit5Dataset(configs.dataset_path, name)
        num_class = 10
        num_client = 4
        data_size = 28
        data_shape = [3, 28, 28]
    elif configs.dataset == 'PACS':
        dataset_names = ['art_painting', 'cartoon', 'photo', 'sketch']
        for name in dataset_names:
            domain_datasets[name] = PACSDataset(configs.dataset_path, name)
        num_class = 7
        num_client = 3
        data_size = 224
        data_shape = [3, 224, 224]
    elif configs.dataset == 'office-10':
        dataset_names = ['amazon', 'caltech', 'dslr', 'webcam']
        for name in dataset_names:
            domain_datasets[name] = Office10Dataset(configs.dataset_path, name)
        num_class = 10
        num_client = 3
        data_size = 300
        data_shape = [3, 300, 300]
    elif configs.dataset == 'domain-net':
        raise NotImplementedError
    else:
        raise ValueError(f'dataset {configs.dataset} not supported')

    if configs.step == 'train':
        train_datasets = []
        for dataset_name in dataset_names:
            if dataset_name == configs.leave_one_out:
                pass
            train_datasets.append(domain_datasets[dataset_name])
        return train_datasets, num_client, num_class, data_size, data_shape
    else:
        return domain_datasets[f'{configs.leave_one_out}'], num_client, num_class, data_size, data_shape
