import logging
import logging
import os

import torch
import wandb
from src.data.data_partition.data_partition import pathological_load_train, dirichlet_load_train
from src.models.resnet import ResNet18
from src.models.wideresnet import WideResNet

from utils.config import parser
from utils.utils import set_seed, make_save_path


# os.environ["HTTPS_PROXY"] = "http://10.162.108.172:7890"

def run():
    configs = parser.parse_args()

    if not configs.debug:
        wandb.init()
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

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(save_path, "output.log"),
        format="[%(asctime)s %(levelname)s] %(message)s",
        filemode="w",
    )

    logging.info(f"-------------------- configuration --------------------")
    for key, value in configs._get_kwargs():
        logging.info(f"configuration {key}: {value}")

    print("prepare dataset...")
    if configs.pathological:
        train_datasets, num_class = pathological_load_train(
            configs.dataset_path, configs.id_dataset, configs.num_client, configs.class_per_client, configs.dataset_seed
        )
    else:
        train_datasets, num_class = dirichlet_load_train(
            configs.dataset_path, configs.id_dataset, configs.num_client, configs.alpha, configs.dataset_seed
        )

    # ---------- construct backbone model ----------
    print("init server and clients...")
    if configs.backbone == "resnet":
        backbone = ResNet18(num_classes=num_class)
    elif configs.backbone == "wideresnet":
        backbone = WideResNet(depth=40, num_classes=num_class, widen_factor=2, dropRate=0.3)
    else:
        raise NotImplementedError("backbone should be ResNet or WideResNet")

    # ---------- construct customized model ----------
    if configs.method.lower() == "fedodg":
        configs.score_model = Energy(net=MLPScore())
        # args.score_model= Energy(net = LatentModel())

    device = torch.device(configs.device)

    if configs.use_score_model:
        configs.score_model = Energy(net=MLPScore())
    # ---------- construct server and clients ----------
    server_args = {
        "join_ratio": configs.join_ratio,
        "checkpoint_path": configs.checkpoint_path,
        "backbone": backbone,
        "device": device,
        "debug": configs.debug,
        "use_score_model": configs.use_score_model,
        "score_model": Energy(net=MLPScore()) if configs.use_score_model else None,
        "alpha": configs.alpha,
        "id_dataset": configs.id_dataset,
    }
    client_args = [
        {
            "cid": cid,
            "device": device,
            "epochs": configs.local_epochs,
            "backbone": backbone,
            "batch_size": configs.batch_size,
            "num_workers": configs.num_workers,
            "pin_memory": configs.pin_memory,
            "train_id_dataset": train_datasets[cid],
            "use_score_model": configs.use_score_model,
        }
        for cid in range(configs.num_client)
    ]

    Server, Client, client_args, server_args = get_server_and_client(configs, client_args, server_args)
    server = Server(server_args)
    clients = [Client(client_args[idx]) for idx in range(configs.num_client)]
    server.clients.extend(clients)

    if configs.method == "FedRoD":
        checkpoint = torch.load(
            f"/root/autodl-tmp/results/{configs.id_dataset}_{configs.alpha}alpha_10clients/FedRoD_wideresnet/model_100.pt"
        )
        server.backbone.load_state_dict(checkpoint["global_net_state_dict"])
        for client_checkpoint, client in zip(checkpoint["clients"], server.clients):
            client.backbone.load_state_dict(checkpoint["global_net_state_dict"])
            client.head.load_state_dict(client_checkpoint["head"])
    if configs.method == "fedodg":
        for client in server.clients:
            client.backbone.load_state_dict(
                torch.load(
                    f"/root/autodl-tmp/results/{configs.id_dataset}_{configs.alpha}alpha_10clients/FedAvg_wideresnet/model_100.pt"
                )
            )
            # client.backbone.load_state_dict(torch.load("/root/autodl-tmp/cifar100_wrn_pretrained_epoch_99.pt"))

    if configs.method == "Ditto":
        checkpoint = torch.load(
            f"/root/autodl-tmp/results/{configs.id_dataset}_{configs.alpha}alpha_10clients/Ditto_wideresnet/model_100.pt"
        )
        server.backbone.load_state_dict(checkpoint["global_net_state_dict"])
        for client_checkpoint, client in zip(checkpoint["clients"], server.clients):
            client.backbone.load_state_dict(checkpoint["global_net_state_dict"])
            client.model_per.load_state_dict(client_checkpoint["client_backbone"])

    # ---------- fit the model ----------
    logging.info("------------------------------ fit the model ------------------------------")
    for t in range(configs.communication_rounds):
        logging.info(f"------------------------- round {t} -------------------------")
        server.fit()
        if (t + 1) % 5 == 0:
            server.make_checkpoint(t)
            # for client in server.clients:
            #     for g in client.score_optimizer.param_groups:
            #         g["lr"] = g["lr"] * 0.1

        # if t == 15:
        #     for client in server.clients:
        #         for g in client.score_optimizer.param_groups:
        #             g["lr"] = g["lr"] * 0.1

        # if (t + 1) % 50 == 0:
        #     server.make_checkpoint(t)

    # ---------- save the model ----------
    print("save the model...")
    server.make_checkpoint(configs.communication_rounds)
    print("done.")


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
