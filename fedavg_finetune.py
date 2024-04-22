import argparse
import logging
import os

import torch

from src.algorithm.FedAvg.client_fedavg import FedAvgClient
from src.algorithm.FedAvg.server_fedavg import FedAvgServer
from src.data.data_partition import load_cifar, load_domains
from src.model.cnn import cnn
from src.model.lenet import lenet
from src.model.resnet import resnet18
from utils.logger import logger
from utils.utils import make_save_path, set_seed

parser = argparse.ArgumentParser(description='arguments for linear probing')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--method', type=str, default='method')
parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'simplecnn', 'shallowcnn', 'lenet'])
parser.add_argument('--task_name', type=str, default='debug_add_loss_0.3')
parser.add_argument('--step', type=str, default='train')
parser.add_argument('--dataset', type=str, default='PACS')
parser.add_argument('--leave_one_out', type=str, default='sketch')
parser.add_argument('--num_client', type=int, default=10, help='number of clients')
parser.add_argument('--alpha', type=float, default=0.1, help='parameter of dirichlet distribution')
parser.add_argument('--dataset_path', type=str, default='/home/yfy/datasets/', help='path to dataset')
parser.add_argument('--num_class', type=int, default=10, help='number of dataset classes')
parser.add_argument('--dataset_seed', type=int, default=21, help='seed to split dataset')
parser.add_argument('--new_dataset_seed', type=int, default=30, help='seed to split dataset')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--join_ratio', type=float, default=1.0, help='join ratio')
parser.add_argument('--global_rounds', type=int, default=100, help='total communication round')
parser.add_argument('--checkpoint_path', type=str, default='default', help='check point path')
parser.add_argument('--epochs', type=int, default=5, help='local epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--model_name', type=str, default='model_round120.pt')
parser.add_argument('--test_batch_size', type=int, default=8)
parser.add_argument('--finetune_model_name', type=str, default='finetune_model.pt')
parser.add_argument('--finetune_epochs', type=int, default=20)


def fedavg_test_cifar(configs, server, checkpoint):
    corrupt_list = ['glass_blur', 'motion_blur', 'contrast', 'impulse_noise', 'gaussian_blur']
    no_shift, label_shift, covariate_shift, hybrid_shift, num_class = load_cifar(configs, corrupt_list)

    def load_checkpoint():
        server.backbone.load_state_dict(checkpoint['backbone'])
        for cid, client_checkpoint in enumerate(checkpoint['clients']):
            server.clients[cid].backbone.load_state_dict(checkpoint['backbone'])
            server.clients[cid].backbone.fc.load_state_dict(client_checkpoint['fc'])

    for name, dataset in zip(['no-shift', 'label_shift'], [no_shift, label_shift]):
        load_checkpoint()
        logger.info(f"test {name} dataset...")
        for cid, client in enumerate(server.clients):
            client.set_test_set(dataset[cid], configs.test_batch_size)
        shift_accuracy = server.test()
        logger.info(f"test {name} dataset accuracy: {shift_accuracy}")

    for name, dataset in zip(['covariate-shift', 'hybrid-shift'], [covariate_shift, hybrid_shift]):
        logger.info(f"test {name} dataset...")
        shift_accuracy = dict()
        for cor_type in corrupt_list:
            shift_accuracy[cor_type] = []
            for severity in range(5):
                load_checkpoint()
                for cid, client in enumerate(server.clients):
                    client.set_test_set(dataset[cor_type][severity][cid], configs.test_batch_size)
                shift_accuracy[cor_type].append(server.test())
        shift_accuracy_mean = {cor_type: sum(shift_accuracy[cor_type]) / 5 for cor_type in corrupt_list}
        logger.info(f"test {name} dataset mean accuracy: {sum(shift_accuracy_mean.values()) / len(corrupt_list)}")
        for cor_type in corrupt_list:
            logger.info(f"test {name} {cor_type} dataset accuracy: {shift_accuracy_mean[cor_type]}")
        for severity in range(5):
            severity_mean_accuracy = sum(shift_accuracy[cor_type][severity] for cor_type in corrupt_list) / len(
                corrupt_list)
            logger.info(f"test {name} severity {severity} mean accuracy: {severity_mean_accuracy}")


def fedavg_test_domain(configs, server):
    test_datasets, num_client, num_class, data_size, data_shape = load_domains(configs)

    logger.info(f"test domain {configs.leave_one_out}...")
    for cid, client in enumerate(server.clients):
        client.set_test_set(test_datasets[cid], configs.test_batch_size)
    accuracy = server.test()
    logger.info(f"test domain {configs.leave_one_out} accuracy: {accuracy}")


def fedavg_test(configs, server):
    checkpoint_path = os.path.join(configs.checkpoint_path, configs.finetune_model_name)
    checkpoint = torch.load(checkpoint_path)

    server.backbone.load_state_dict(checkpoint['backbone'])
    for cid, client_checkpoint in enumerate(checkpoint['clients']):
        server.clients[cid].backbone.load_state_dict(checkpoint['backbone'])
        server.clients[cid].backbone.fc.load_state_dict(client_checkpoint['fc'])

    setattr(configs, "step", 'test')

    if configs.dataset in ['cifar10', 'cifar100']:
        fedavg_test_cifar(configs, server, checkpoint)
    else:
        fedavg_test_domain(configs, server)


def finetune(configs, server):
    checkpoint_path = os.path.join(configs.checkpoint_path, configs.model_name)
    checkpoint = torch.load(checkpoint_path)
    if configs.method == 'fedavg':
        server.load_checkpoint(checkpoint)
    elif configs.method == 'fedicon':
        server.backbone.load_state_dict(checkpoint['backbone'])
        for cid, client_checkpoint in enumerate(checkpoint['clients']):
            server.clients[cid].backbone.load_state_dict(checkpoint['backbone'])
            server.clients[cid].backbone.fc.load_state_dict(client_checkpoint['fc'])
    else:
        raise ValueError("method unavailable")

    for client in server.clients:
        client.backbone.to(client.device)
        client.backbone.eval()

        optimizer = torch.optim.SGD(
            client.backbone.fc.parameters(),
            lr=client.learning_rate,
            momentum=client.momentum,
            weight_decay=client.weight_decay,
        )

        for _ in range(configs.finetune_epochs):
            for data, target in client.train_loader:
                data, target = data.to(client.device), target.to(client.device)
                logit = client.backbone(data)
                loss = torch.nn.functional.cross_entropy(logit, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        client.backbone.cpu()

    finetune_checkpoint_name = os.path.join(configs.checkpoint_path, configs.finetune_model_name)
    if configs.method == 'fedavg':
        torch.save(
            {
                'backbone': checkpoint,
                'clients': [{'fc': client.backbone.fc.state_dict()} for client in server.clients],
            },
            finetune_checkpoint_name,
        )
    elif configs.method == 'fedicon':
        torch.save(
            {
                'backbone': checkpoint['backbone'],
                'clients': [{'fc': client.backbone.fc.state_dict()} for client in server.clients]
            },
            finetune_checkpoint_name,
        )
    else:
        raise ValueError("method unavailable")


def prepare_model():
    configs = parser.parse_args()
    set_seed(configs.seed)
    save_path = make_save_path(configs)
    if configs.checkpoint_path == "default":
        setattr(configs, "checkpoint_path", save_path)
    file_handler = logging.FileHandler(os.path.join(save_path, 'finetune.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"-------------------- configuration --------------------")
    for key, value in vars(configs).items():
        logger.info(f"configuration {key}: {value}")

    logger.info("prepare dataset...")
    if configs.dataset in ['cifar10', 'cifar100']:
        train_datasets, num_class = load_cifar(configs)
        data_size = 32
        data_shape = [3, 32, 32]
    elif configs.dataset in ['digit-5', 'PACS', 'office-10', 'domain-net']:
        train_datasets, num_client, num_class, data_size, data_shape = load_domains(configs)
        setattr(configs, "num_client", num_client)
        logger.info(f"update configuration num_client: {num_client}")
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

    logger.info("prepare server and clients...")
    device = torch.device(configs.device)
    server = FedAvgServer(device, backbone, configs)
    clients = [FedAvgClient(cid, device, backbone, configs) for cid in range(configs.num_client)]
    for cid, client in enumerate(clients):
        client.set_train_set(train_datasets[cid])
    server.clients.extend(clients)

    if configs.step == 'train':
        finetune(configs, server)

    if configs.method == 'fedavg':
        fedavg_test(configs, server)

    logger.info("done")


if __name__ == '__main__':
    prepare_model()
