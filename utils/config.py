import argparse

parser = argparse.ArgumentParser(description='arguments for training and testing')

parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--use_profile', type=bool, default=False)
parser.add_argument('--wandb_mode', type=str, choices=['disabled', 'online', 'offline'], default='online')
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--method', type=str, default='fediir')
parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'simplecnn', 'shallowcnn', 'lenet'])
parser.add_argument('--task_name', type=str, default='default_setting')
parser.add_argument('--step', type=str, default='train')
# ---------- dataset partition ----------
parser.add_argument('--dataset', type=str, default='PACS')
parser.add_argument('--leave_one_out', type=str, default='cartoon')
parser.add_argument('--num_client', type=int, default=10, help='number of clients')
parser.add_argument('--alpha', type=float, default=0.1, help='parameter of dirichlet distribution')
parser.add_argument('--dataset_path', type=str, default='/home/yfy/datasets/', help='path to dataset')
parser.add_argument('--num_class', type=int, default=10, help='number of dataset classes')
parser.add_argument('--dataset_seed', type=int, default=21, help='seed to split dataset')
parser.add_argument('--new_dataset_seed', type=int, default=30, help='seed to split dataset')
# ---------- device ----------
parser.add_argument('--device', type=str, default='cuda:1', help='device')
# ---------- server configuration ----------
parser.add_argument('--join_ratio', type=float, default=1.0, help='join ratio')
parser.add_argument('--global_rounds', type=int, default=50, help='total communication round')
parser.add_argument('--checkpoint_path', type=str, default='default', help='check point path')
# ---------- client configuration ----------
parser.add_argument('--epochs', type=int, default=5, help='local epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
# ---------- optimizer --------
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=5e-4)
# ---------- test --------
parser.add_argument('--model_name', type=str, default='model_round100.pt')
parser.add_argument('--test_batch_size', type=int, default=8)

# ---------- fedicon configuration --------
parser.add_argument('--icon_rounds', type=int, default=100)
parser.add_argument('--icon_learning_rate', type=float, default=0.005)
parser.add_argument('--finetune_rounds', type=int, default=1)
parser.add_argument('--finetune_method', type=str, default='per')
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--unsupervised_temperature', type=float, default=0.1)
parser.add_argument('--finetune_epochs', type=int, default=20)

# ---------- fedthe configuration --------
parser.add_argument('--personal_head_epoch', type=int, default=1)
parser.add_argument('--e_learning_rate', type=float, default=0.1)
parser.add_argument('--the_alpha', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.3)

# ---------- method configuration --------
parser.add_argument('--avg_head', type=bool, default=False)
parser.add_argument('--aux_ratio', type=float, default=1.0)
parser.add_argument('--trade_off', type=float, default=0.3)
parser.add_argument('--add_loss', type=bool, default=True)
parser.add_argument('--loss_weight', type=float, default=0.3)
parser.add_argument('--try_method', type=str, default='feature_aux-classifier')

# ---------- atp configuration --------
parser.add_argument('--load_model_name', type=str, default='model_round100.pt')

# ---------- fediir configuration --------
parser.add_argument('--fediir_ema', type=float, default=0.95)
parser.add_argument('--penalty_weight', type=float, default=1e-3)