import os

import lightning as L
from torch.utils.data import DataLoader, Dataset

import datasets.transforms as T
from util.misc import ParserObject
from util.file_ok import file_ok

class AbstractDataSet(Dataset):
    def __init__(self, root, split):
        self.root = root

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class AbstractDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        for i, j in config.items():
            self.__dict__[i] = j

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def __getattr__(self, key):
        if key in self.__dict__:
            return self[key]

    def prepare_data(self):
        for data_path, expect in self.check_table.items():
            data_path=os.path.join(self.root, 'processed',self.dataset_name, data_path)
            if not os.path.exists(data_path) or not file_ok(data_path,expect):
                answer=input(f'{data_path} is not exist or incorrect, do you want to re-process it? [y/n]')
                if answer=='y':
                    self.extract_data(data_path)
                else:
                    raise NotImplemented(f'Experiment stopped due to the incorrect data')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
