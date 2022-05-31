import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import pytorch_lightning as pl


class CrossSubjectDataModule(pl.LightningDataModule):
    def __init__(self, test_subject, X: torch.FloatTensor, labels: torch.LongTensor, metadata, dataloader_kwargs=None):
        super().__init__()
        self.test_subject = test_subject
        self.X = X
        self.labels = labels
        self.metadata = metadata
        self.dataloader_kwargs = dataloader_kwargs if dataloader_kwargs is not None else dict()

        # setup:
        self.dataset = TensorDataset(self.X, self.labels)
        if self.test_subject not in self.metadata.subject:
            raise ValueError(f'Test subject {self.test_subject} missing from metadata')
        mask_test = (self.metadata.subject==self.test_subject).to_numpy()
        self.ids_test = np.arange(len(labels))[mask_test]
        self.ids_train = np.arange(len(labels))[~mask_test]

    def train_dataloader(self):
        return DataLoader(Subset(self.dataset, self.ids_train), **self.dataloader_kwargs, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(Subset(self.dataset, self.ids_val), **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(Subset(self.dataset, self.ids_test), **self.dataloader_kwargs)
