import logging
import os

import torch

from src.data.data_partition import load_cifar, load_domains
from src.model.cnn import cnn
from src.model.lenet import lenet
from src.model.resnet import resnet18
from utils.config import parser
from utils.logger import logger
from utils.utils import set_seed, make_save_path, prepare_server_and_clients


def test_cifar(configs):
    corrupt_list = ['glass_blur', 'motion_blur', 'contrast', 'impulse_noise', 'gaussian_blur']
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
    server_object, client_object = prepare_server_and_clients(configs)
    device = torch.device(configs.device)
    server = server_object(device, backbone, configs, None)
    clients = [client_object(cid, device, backbone, configs) for cid in range(configs.num_client)]
    server.clients.extend(clients)

    checkpoint_path = os.path.join(configs.checkpoint_path, configs.model_name)
    checkpoint = torch.load(checkpoint_path)

    for name, dataset in zip(['no-shift', 'label_shift'], [no_shift, label_shift]):
        server.load_checkpoint(checkpoint)
        logger.info(f"test {name} dataset...")
        for cid, client in enumerate(server.clients):
            client.set_test_set(dataset[cid], configs.test_batch_size)
        plain_shift_accuracy = server.plain_test()
        logger.info(f"plain-test {name} dataset accuracy: {plain_shift_accuracy}")
        shift_accuracy = server.test()
        logger.info(f"test {name} dataset accuracy: {shift_accuracy}")

    for name, dataset in zip(['covariate-shift', 'hybrid-shift'], [covariate_shift, hybrid_shift]):
        logger.info(f"test {name} dataset...")
        plain_shift_accuracy = dict()
        shift_accuracy = dict()
        for cor_type in corrupt_list:
            plain_shift_accuracy[cor_type] = []
            shift_accuracy[cor_type] = []
            for severity in range(5):
                server.load_checkpoint(checkpoint)
                for cid, client in enumerate(server.clients):
                    client.set_test_set(dataset[cor_type][severity][cid], configs.test_batch_size)
                plain_shift_accuracy[cor_type].append(server.plain_test())
                shift_accuracy[cor_type].append(server.test())
        plain_shift_accuracy_mean = {cor_type: sum(plain_shift_accuracy[cor_type]) / 5 for cor_type in corrupt_list}
        logger.info(
            f"plain-test {name} dataset mean accuracy: {sum(plain_shift_accuracy_mean.values()) / len(corrupt_list)}")
        shift_accuracy_mean = {cor_type: sum(shift_accuracy[cor_type]) / 5 for cor_type in corrupt_list}
        logger.info(f"test {name} dataset mean accuracy: {sum(shift_accuracy_mean.values()) / len(corrupt_list)}")
        for cor_type in corrupt_list:
            logger.info(f"plain-test {name} {cor_type} dataset accuracy: {plain_shift_accuracy_mean[cor_type]}")
            logger.info(f"test {name} {cor_type} dataset accuracy: {shift_accuracy_mean[cor_type]}")
        for severity in range(5):
            plain_severity_mean_accuracy = sum(
                plain_shift_accuracy[cor_type][severity] for cor_type in corrupt_list) / len(corrupt_list)
            logger.info(f"plain-test {name} severity {severity} mean accuracy: {plain_severity_mean_accuracy}")
            severity_mean_accuracy = sum(shift_accuracy[cor_type][severity] for cor_type in corrupt_list) / len(
                corrupt_list)
            logger.info(f"test {name} severity {severity} mean accuracy: {severity_mean_accuracy}")


def test_domain(configs):
    test_datasets, num_client, num_class, data_size, data_shape = load_domains(configs)
    setattr(configs, "num_client", num_client)

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
    server_object, client_object = prepare_server_and_clients(configs)
    device = torch.device(configs.device)
    server = server_object(device, backbone, configs)
    clients = [client_object(cid, device, backbone, configs) for cid in range(configs.num_client)]
    server.clients.extend(clients)

    checkpoint_path = os.path.join(configs.checkpoint_path, configs.model_name)
    checkpoint = torch.load(checkpoint_path)

    server.load_checkpoint(checkpoint)
    logger.info(f"test domain {configs.leave_one_out}...")
    for cid, client in enumerate(server.clients):
        client.set_test_set(test_datasets[cid], configs.test_batch_size)
    plain_accuracy = server.plain_test()
    logger.info(f"plain-test domain {configs.leave_one_out} accuracy: {plain_accuracy}")
    accuracy = server.test()
    logger.info(f"test domain {configs.leave_one_out} accuracy: {accuracy}")


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
        test_domain(configs)

    logger.info("done")


if __name__ == '__main__':
    test()
