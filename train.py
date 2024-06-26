import logging
import os

import torch
import wandb
from torch.profiler import profile, tensorboard_trace_handler, schedule, ProfilerActivity

from src.data.data_partition import load_cifar, load_domains
from src.model.cnn import cnn
from src.model.lenet import lenet
from src.model.resnet import resnet18
from utils.config import parser
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
    if not configs.debug:
        wandb.config.update(configs, allow_val_change=True)

    profiler = None
    if configs.use_profile:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=0, active=1, repeat=configs.global_rounds),
            on_trace_ready=tensorboard_trace_handler(os.path.join(save_path, 'profiler')),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()

    file_handler = logging.FileHandler(os.path.join(save_path, 'logfile.log'))
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
        setattr(configs, "num_class", num_class)
        logger.info(f"update configuration num_class: {num_class}")
    elif configs.dataset in ['digit-5', 'PACS', 'office-10', 'domain-net']:
        train_datasets, num_client, num_class, data_size, data_shape = load_domains(configs)
        setattr(configs, "num_client", num_client)
        setattr(configs, "num_class", num_class)
        logger.info(f"update configuration num_class: {num_class}")
        logger.info(f"update configuration num_client: {num_client}")
        if not configs.debug:
            wandb.config.update(configs, allow_val_change=True)
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

    if configs.method == 'atp':
        checkpoint_path = os.path.join(configs.checkpoint_path, configs.load_model_name)
        checkpoint = torch.load(checkpoint_path)
        backbone.load_state_dict(checkpoint)

    logger.info("prepare server and clients...")
    server_object, client_object = prepare_server_and_clients(configs)
    device = torch.device(configs.device)
    server = server_object(device, backbone, configs, profiler)
    clients = [client_object(cid, device, backbone, configs) for cid in range(configs.num_client)]
    for cid, client in enumerate(clients):
        client.set_train_set(train_datasets[cid])
    server.clients.extend(clients)

    server.fit()

    logger.info("done")

    if configs.use_profile:
        profiler.stop()


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

        if args.dataset in ['cifar10', 'cifar100']:
            project_name = f"{args.dataset}_{args.num_client}clients_alpha{args.alpha}_fedtta"
        else:
            project_name = f"{args.dataset}_leave-{args.leave_one_out}"
        sweep_id = wandb.sweep(
            sweep_configuration,
            project=project_name,
        )
        wandb.agent(sweep_id, function=run)
