import logging
import os

import torch
import torch.nn.functional as F

from src.data.data_partition import load_cifar
from src.model.cnn import cnn
from src.model.lenet import lenet
from src.model.resnet import resnet18
from utils.config import parser
from utils.logger import logger
from utils.utils import set_seed, make_save_path, prepare_server_and_clients


def finetune_and_test(configs, server, train_datasets, no_shift, label_shift, covariate_shift, hybrid_shift, corrupt_list):
    for cid, client in enumerate(server.clients):
        client.set_train_set(train_datasets[cid])
        client.backbone.to(client.device)
        client.backbone.train()
        finetune_optimizer = torch.optim.SGD(
            params=client.backbone.fc.parameters(),
            lr=0.1,
        )

        for finetune_epoch in range(20):
            for data, target in client.train_loader:
                data, target = data.to(client.device), target.to(client.device)
                logits = client.backbone(data)
                loss = F.cross_entropy(logits, target)
                finetune_optimizer.zero_grad()
                loss.backward()
                finetune_optimizer.step()
        client.backbone.cpu()

    logger.info("test no-shift dataset...")
    for cid, client in enumerate(server.clients):
        client.set_test_set(no_shift[cid], configs.test_batch_size)
    plain_no_shift_accuracy = server.plain_test()
    logger.info(f"plain-test no-shift dataset accuracy: {plain_no_shift_accuracy}")
    no_shift_accuracy = server.test()
    logger.info(f"test no-shift dataset accuracy: {no_shift_accuracy}")

    logger.info("test label-shift dataset...")
    for cid, client in enumerate(server.clients):
        client.set_test_set(label_shift[cid], configs.test_batch_size)
    plain_label_shift_accuracy = server.plain_test()
    logger.info(f"plain-test label-shift dataset accuracy: {plain_label_shift_accuracy}")
    label_shift_accuracy = server.test()
    logger.info(f"test label-shift dataset accuracy: {label_shift_accuracy}")

    for name, dataset in zip(['covariate-shift', 'hybrid-shift'], [covariate_shift, hybrid_shift]):
        logger.info(f"test {name} dataset...")
        plain_shift_accuracy = dict()
        shift_accuracy = dict()
        for cor_type in corrupt_list:
            plain_shift_accuracy[cor_type] = []
            shift_accuracy[cor_type] = []
            for severity in range(5):
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

def vis_cifar(configs):
    corrupt_list = ['glass_blur', 'motion_blur', 'contrast', 'impulse_noise', 'gaussian_blur']
    no_shift, label_shift, covariate_shift, hybrid_shift, num_class = load_cifar(configs, corrupt_list)
    setattr(configs, 'step', 'train')
    train_set, _ = load_cifar(configs)
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
    server.load_checkpoint(checkpoint)


def visualization():
    configs = parser.parse_args()
    set_seed(configs.seed)
    save_path = make_save_path(configs)
    if configs.checkpoint_path == "default":
        setattr(configs, "checkpoint_path", save_path)

    file_handler = logging.FileHandler(os.path.join(save_path, 'visfile.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"-------------------- configuration --------------------")
    for key, value in vars(configs).items():
        logger.info(f"configuration {key}: {value}")

    logger.info("prepare dataset...")
    if configs.dataset in ['cifar10', 'cifar100']:
        vis_cifar(configs)
    elif configs.dataset in ['digit-5', 'PACS', 'office-10', 'domain-net']:
        pass

    logger.info("done")


if __name__ == '__main__':
    visualization()
