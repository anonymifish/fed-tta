from torch.utils.data import Dataset


class AugPairDataset(Dataset):
    def __init__(self, dataset, transform, supervise=True):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.supervise = supervise

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        if self.supervise:
            return self.transform(x), self.transform(x), y
        else:
            return x, self.transform(x), y

    def __len__(self) -> int:
        return len(self.dataset)