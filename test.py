import logging
import os

import torch

from src.data.data_partition import load_cifar
from src.model.cnn import cnn
from src.model.lenet import lenet
from src.model.resnet import resnet18
from utils.config import parser
from utils.logger import logger
from utils.utils import set_seed, make_save_path, prepare_server_and_clients


def test_cifar(configs):
    corrupt_list = []
    no_shift, label_shift, covariate_shift, hybrid_shift, num_class = load_cifar(configs, corrupt_list)
    data_size = 32
    data_shape = [3, 32, 32]

    logger.info("init server and clients...")
    if configs.backbone == 'lenet':
        backbone = lenet(data_size, num_class)
    elif configs.backbone in ['simplecnn', 'shallowcnn']:
        backbone = cnn(configs.backbone, data_shape, num_class)
    elif configs.backbone == "resnet":
        backbone = resnet18(num_classes=num_class)
    else:
        raise ValueError("backbone unavailable")

    logger.info("prepare server and clients...")
    server_object, client_object = prepare_server_and_clients(backbone, configs)
    device = torch.device(configs.device)
    server = server_object(device, backbone, configs)
    clients = [client_object(cid, device, backbone, configs) for cid in range(configs.num_client)]
    server.clients.extend(clients)

    checkpoint_path = os.path.join(configs.checkpoint_path, configs.model_name)
    checkpoint = torch.load(checkpoint_path)

    server.load_state_dict(checkpoint)
    logger.info("test no shift dataset...")
    for cid, client in enumerate(server.clients):
        client.set_test_set(no_shift[cid], configs.test_batch_size)
    no_shift_accuracy = server.test()
    logger.info(f"test no shift dataset accuracy: {no_shift_accuracy}")

    server.load_state_dict(checkpoint)
    logger.info("test label shift dataset...")
    for cid, client in enumerate(server.clients):
        client.set_test_set(label_shift[cid], configs.test_batch_size)
    label_shift_accuracy = server.test()
    logger.info(f"test label shift dataset accuracy: {label_shift_accuracy}")

    server.load_state_dict(checkpoint)
    logger.info("test covariate shift dataset...")
    covariate_shift_accuracy = dict()
    for cor_type in corrupt_list:
        covariate_shift_accuracy[cor_type] = []
        for severity in range(5):
            for cid, client in enumerate(server.clients):
                client.set_test_set(covariate_shift[cor_type][severity][cid], configs.test_batch_size)
            covariate_shift_accuracy[cor_type].append(server.test())
    covariate_shift_accuracy_mean = {cor_type: sum(covariate_shift_accuracy[cor_type]) / 5 for cor_type in corrupt_list}
    logger.info(
        f"test covariate shift dataset mean accuracy: {sum(covariate_shift_accuracy_mean.values()) / len(corrupt_list)}")
    for cor_type in corrupt_list:
        logger.info(f"test covariate shift {cor_type} dataset accuracy: {covariate_shift_accuracy_mean[cor_type]}")
    for severity in range(5):
        severity_mean_accuracy = sum([covariate_shift_accuracy][cor_type][severity] for cor_type in corrupt_list) / len(
            corrupt_list)
        logger.info(f"test covariate shift {severity} mean accuracy: {severity_mean_accuracy}")

    server.load_state_dict(checkpoint)
    logger.info("test hybrid shift dataset...")
    hybrid_shift_accuracy = dict()
    for cor_type in corrupt_list:
        hybrid_shift_accuracy[cor_type] = []
        for severity in range(5):
            for cid, client in enumerate(server.clients):
                client.set_test_set(covariate_shift[cor_type][severity][cid], configs.test_batch_size)
            hybrid_shift_accuracy[cor_type].append(server.test())
    hybrid_shift_accuracy_mean = {cor_type: sum(hybrid_shift_accuracy[cor_type]) / 5 for cor_type in corrupt_list}
    logger.info(
        f"test covariate shift dataset mean accuracy: {sum(hybrid_shift_accuracy_mean.values()) / len(corrupt_list)}")
    for cor_type in corrupt_list:
        logger.info(f"test covariate shift {cor_type} dataset accuracy: {hybrid_shift_accuracy_mean[cor_type]}")
    for severity in range(5):
        severity_mean_accuracy = sum([hybrid_shift_accuracy][cor_type][severity] for cor_type in corrupt_list) / len(
            corrupt_list)
        logger.info(f"test covariate shift {severity} mean accuracy: {severity_mean_accuracy}")


def test():
    configs = parser.parse_args()
    set_seed(configs.seed)
    save_path = make_save_path(configs)
    if configs.checkpoint_path == "default":
        setattr(configs, "checkpoint_path", save_path)

    file_handler = logging.FileHandler(os.path.join(save_path, 'testfile.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"-------------------- configuration --------------------")
    for key, value in vars(configs).items():
        logger.info(f"configuration {key}: {value}")

    logger.info("prepare dataset...")
    if configs.dataset in ['cifar10', 'cifar100']:
        test_cifar(configs)
    elif configs.dataset in ['digit-5', 'PACS', 'office-10', 'domain-net']:
        pass

    logger.info("done")


if __name__ == '__main__':
    test()
