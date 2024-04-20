import argparse
import logging
import os

import torch

from src.data.data_partition import load_cifar, load_domains
from src.model.cnn import cnn
from src.model.lenet import lenet
from src.model.resnet import resnet18
from utils.logger import logger
from utils.utils import set_seed, make_save_path

parser = argparse.ArgumentParser(description='arguments for linear probing')


def train_linear_and_test(backbone, test_set):
    with torch.no_grad():
        pass


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
    logger.info("prepare dataset...")
    if configs.dataset in ['cifar10', 'cifar100']:
        no_shift, label_shift, covariate_shift, hybrid_shift, num_class = load_cifar(configs, corrupt_list)
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

    if configs.dataset in ['cifar10', 'cifar100']:
        pass
    else:
        pass


if __name__ == '__main__':
    linear_probing()
