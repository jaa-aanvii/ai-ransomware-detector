import torch
from torch.utils.data import Dataset, DataLoader


class RansomwareSequenceDataset(Dataset):

    def __init__(
        self,
        sequences,
        labels
    ):
        self.X = torch.tensor(
            sequences,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            labels,
            dtype=torch.float32
        ).view(-1, 1)

        if self.X.ndim != 3:
            raise ValueError(
                f"Expected 3D input, got {self.X.shape}"
            )

        if self.X.shape[1] != 6:
            raise ValueError(
                f"Expected sequence length 6, "
                f"got {self.X.shape[1]}"
            )

        if self.X.shape[2] != 8:
            raise ValueError(
                f"Expected 8 features, "
                f"got {self.X.shape[2]}"
            )

        if len(self.X) != len(self.y):
            raise ValueError(
                "Number of sequences and labels must match"
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def create_dataloader(
    sequences,
    labels,
    batch_size=16,
    shuffle=True
):
    dataset = RansomwareSequenceDataset(
        sequences,
        labels
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )