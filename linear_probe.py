import argparse
import logging
import os

import torch
from sklearn.metrics import accuracy_score
from tensorloader import TensorLoader
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.data_partition import load_cifar, load_domains
from src.model.cnn import cnn
from src.model.lenet import lenet
from src.model.resnet import resnet18
from utils.logger import logger
from utils.utils import set_seed, make_save_path

parser = argparse.ArgumentParser(description='arguments for linear probing')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--method', type=str, default='fedcal')
parser.add_argument('--backbone', type=str, default='lenet', choices=['resnet', 'simplecnn', 'shallowcnn', 'lenet'])
parser.add_argument('--task_name', type=str, default='debug')
parser.add_argument('--step', type=str, default='test')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--leave_one_out', type=str, default='cartoon')
parser.add_argument('--num_client', type=int, default=10, help='number of clients')
parser.add_argument('--alpha', type=float, default=0.1, help='parameter of dirichlet distribution')
parser.add_argument('--dataset_path', type=str, default='/home/yfy/datasets/', help='path to dataset')
parser.add_argument('--num_class', type=int, default=10, help='number of dataset classes')
parser.add_argument('--dataset_seed', type=int, default=21, help='seed to split dataset')
parser.add_argument('--new_dataset_seed', type=int, default=30, help='seed to split dataset')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--checkpoint_path', type=str, default='default', help='check point path')
parser.add_argument('--epochs', type=int, default=200, help='linear probe epochs')
parser.add_argument('--model_name', type=str, default='model_round100.pt')


def train_linear_and_test(backbone, device, dataset, epochs):
    train_set, test_set = random_split(dataset, [int(0.7 * len(dataset)), len(dataset) - int(0.7 * len(dataset))])

    backbone.to(device)
    backbone.eval()
    with torch.no_grad():
        train_feature = torch.concat([
            backbone.intermediate_forward(data.to(device)).cpu() for data, _ in
            DataLoader(train_set, 128, shuffle=False)
        ])
        train_label = torch.concat([target for _, target in DataLoader(train_set, 128, shuffle=False)])
        test_feature = torch.concat([
            backbone.intermediate_forward(data.to(device)).cpu() for data, _ in DataLoader(test_set, 128, shuffle=False)
        ])
        test_label = torch.concat([target for _, target in DataLoader(test_set, 128, shuffle=False)])
    backbone.cpu()

    classifier = torch.nn.Linear(backbone.fc.in_features, backbone.fc.out_features)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    classifier.to(device)
    classifier.train()
    for _ in tqdm(range(epochs), desc=f'train linear'):
        for feature, label in TensorLoader((train_feature, train_label), batch_size=128, shuffle=True):
            feature, label = feature.to(device), label.to(device)
            logit = classifier(feature)
            loss = torch.nn.functional.cross_entropy(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    classifier.eval()
    test_pred = torch.concat([
        classifier(feature.to(device)).argmax(dim=1).cpu() for feature in TensorLoader(test_feature, batch_size=128)
    ])
    acc = accuracy_score(test_label, test_pred)
    return acc


def linear_probing():
    configs = parser.parse_args()
    set_seed(configs.seed)
    save_path = make_save_path(configs)
    if configs.checkpoint_path == "default":
        setattr(configs, "checkpoint_path", save_path)
    file_handler = logging.FileHandler(os.path.join(save_path, 'linear_probe.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"-------------------- configuration --------------------")
    for key, value in vars(configs).items():
        logger.info(f"configuration {key}: {value}")

    logger.info("prepare dataset...")
    corrupt_list = ['glass_blur', 'motion_blur', 'contrast', 'impulse_noise', 'gaussian_blur']
    if configs.dataset in ['cifar10', 'cifar100']:
        _, no_shift, label_shift, covariate_shift, hybrid_shift, num_class = load_cifar(configs, corrupt_list)
        data_size = 32
        data_shape = [3, 32, 32]
    elif configs.dataset in ['digit-5', 'PACS', 'office-10', 'domain-net']:
        test_datasets, num_client, num_class, data_size, data_shape = load_domains(configs)
        setattr(configs, "num_client", num_client)
    else:
        raise ValueError('illegal dataset')

    logger.info("init server and clients...")
    if configs.backbone == 'lenet':
        backbone = lenet(data_size, num_class)
    elif configs.backbone in ['simplecnn', 'shallowcnn']:
        backbone = cnn(configs.backbone, data_shape, num_class)
    elif configs.backbone == "resnet":
        backbone = resnet18(num_classes=num_class)
    else:
        raise ValueError("backbone unavailable")

    device = torch.device(configs.device)

    checkpoint_path = os.path.join(configs.checkpoint_path, configs.model_name)
    checkpoint = torch.load(checkpoint_path)
    if configs.method in ['fedavg', 'fedcal']:
        backbone.load_state_dict(checkpoint)
    else:
        backbone.load_state_dict(checkpoint['backbone'])

    if configs.dataset in ['cifar10', 'cifar100']:
        for name, dataset in zip(['no-shift', 'label-shift'], [no_shift, label_shift]):
            logger.info(f"linear probe on {name} dataset...")
            acc_list = []
            weights = [len(_) for _ in dataset]
            for cid in range(len(dataset)):
                acc_list.append(train_linear_and_test(backbone, device, dataset[cid], configs.epochs))
            acc = sum([acc_list[i] * weights[i] / sum(weights) for i in range(len(dataset))])
            logger.info(f"linear probe on {name} dataset accuracy = {acc}")

        for name, dataset in zip(['covariate-shift', 'hybrid-shift'], [covariate_shift, hybrid_shift]):
            for cor_type in corrupt_list:
                logger.info(f"linear probe on {name}, {cor_type} shift dataset...")
                acc_list = []
                weights = [len(_) for _ in dataset[cor_type][4]]
                for cid in range(len(dataset[cor_type][4])):
                    acc_list.append(train_linear_and_test(backbone, device, dataset[cor_type][4][cid], configs.epochs))
                acc = sum([acc_list[i] * weights[i] / sum(weights) for i in range(len(dataset[cor_type][4]))])
                logger.info(f"linear probe on {name}, {cor_type} shift, severity = 4 dataset accuracy = {acc}")
    else:
        linear_probe_set = test_datasets[0]
        acc = train_linear_and_test(backbone, device, linear_probe_set, configs.epochs)
        logger.info(f"linear probe on domain {configs.leave_one_out} accuracy = {acc}")

    logger.info("done")


if __name__ == '__main__':
    linear_probing()
