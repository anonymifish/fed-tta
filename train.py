import logging

import torch
import wandb

from src.data.data_partition import load_cifar
from src.model.cnn import cnn
from src.model.lenet import lenet
from src.model.resnet import resnet18
from utils.config import parser
from utils.logger import change_file_path
from utils.logger import logger
from utils.utils import set_seed, make_save_path, prepare_server_and_clients


# os.environ["HTTPS_PROXY"] = "http://10.162.108.172:7890"

def run():
    configs = parser.parse_args()

    if not configs.debug:
        wandb.init(mode=configs.wandb_mode)
        for key in dict(wandb.config):
            setattr(configs, key, dict(wandb.config)[key])
        wandb.config.update(configs)
        wandb.run.name = (
            f"{configs.method}_{configs.backbone}_{configs.task_name}"
        )

    set_seed(configs.seed)
    save_path = make_save_path(configs)
    if configs.checkpoint_path == "default":
        setattr(configs, "checkpoint_path", save_path)
    wandb.config.update(configs)

    change_file_path(save_path)

    logger.info(f"-------------------- configuration --------------------")
    for key, value in configs._get_kwargs():
        logging.info(f"configuration {key}: {value}")

    logger.info("prepare dataset...")
    if configs.dataset in ['cifar10', 'cifar100']:
        train_datasets, num_class = load_cifar(configs)
        data_size = 32
        data_shape = [3, 32, 32]
    elif configs.dataset in ['digit-5', 'PACS', 'office-10', 'domain-net']:
        pass

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
    clients = [client_object(cid, device, backbone, configs) for cid in range(configs.num_clients)]
    for cid, client in enumerate(clients):
        client.set_train_set(train_datasets[cid])
    server.clients.extend(clients)

    server.fit()

    logger.info("done")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.debug:
        run()
    else:
        sweep_configuration = {
            "method": "grid",
            "parameters": {
                "learning_rate": {
                    "values": [args.learning_rate],
                },
            },
        }

        sweep_id = wandb.sweep(
            sweep_configuration,
            project=f"{args.dataset}_{args.num_client}clients_alpha{args.alpha}_fedtta",
        )
        wandb.agent(sweep_id, function=run)
