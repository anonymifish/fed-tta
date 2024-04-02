import copy

from torch.utils.data import DataLoader


class BaseClient:

    def __init__(self, cid, device, backbone, configs):
        self.cid = cid
        self.device = device
        self.backbone = copy.deepcopy(backbone)
        self.epochs = configs.epochs
        self.learning_rate = configs.learning_rate
        self.momentum = configs.momentum
        self.weight_decay = configs.weight_decay
        self.batch_size = configs.batch_size
        self.debug = configs.debug

        self.train_set = None
        self.test_set = None
        self.test_batch_size = None
        self.train_dataloader = None
        self.test_dataloader = None

    def set_train_set(self, train_set):
        self.train_set = train_set
        self.train_dataloader = DataLoader(
            dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=3, pin_memory=False,
        )

    def set_test_set(self, test_set, test_batch_size):
        self.test_set = test_set
        self.test_batch_size = test_batch_size
        self.test_dataloader = DataLoader(
            dataset=self.train_set, batch_size=self.test_batch_size, shuffle=True, num_workers=3, pin_memory=False,
        )
