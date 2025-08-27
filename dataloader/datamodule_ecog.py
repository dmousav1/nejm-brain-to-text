import os
import random
import re
import string
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import lightning as pl
from torchvision import transforms


class RNNTMEADataset(Dataset):
    """
    Reads MEA trials from native HDF5 files and returns RNNT-ready fields.
    Each item is a single trial: inputs [T, C], input_length, targets [U], target_length.
    """

    def __init__(
        self,
        examples: List[Dict],
        feature_subset: Optional[List[int]] = None,
        transform=None,
    ):
        super().__init__()
        self.examples = examples
        self.feature_subset = feature_subset
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        with h5py.File(ex["session_path"], "r") as f:
            g = f[f"trial_{ex['trial_idx']:04d}"]
            x = g["input_features"][:]
            y = g["seq_class_ids"][:]
            block_num = g.attrs.get("block_num", -1)
            trial_num = g.attrs.get("trial_num", -1)
        if self.feature_subset is not None and len(self.feature_subset) > 0:
            x = x[:, self.feature_subset]
        x = torch.from_numpy(x).float()
        if self.transform is not None:
            x = self.transform(x)
        y = torch.from_numpy(y).long()
        item = {
            "inputs": x,
            "input_length": torch.tensor(len(x), dtype=torch.long),
            "targets": y,
            "target_length": torch.tensor(len(y), dtype=torch.long),
            "day_index": torch.tensor(ex["day_idx"], dtype=torch.long),
            "block_num": torch.tensor(block_num, dtype=torch.long),
            "trial_num": torch.tensor(trial_num, dtype=torch.long),
        }
        return item

    @staticmethod
    def collate(batch):
        inputs = [b["inputs"] for b in batch]
        input_lengths = torch.tensor([x.shape[0] for x in inputs], dtype=torch.long)
        inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0.0)
        targets = [b["targets"] for b in batch]
        target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
        return {
            "inputs": inputs,
            "input_lengths": input_lengths,
            "targets": targets,
            "target_lengths": target_lengths,
            "day_indices": torch.stack([b["day_index"] for b in batch]),
            "block_nums": torch.stack([b["block_num"] for b in batch]),
            "trial_nums": torch.stack([b["trial_num"] for b in batch]),
        }


class ECoGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir,
        sessions: List[str],
        trainval_ratio: List[float] = [0.95, 0.05],
        shuffle_trainval_split: bool = False,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        transform_config: Dict = {},
        num_workers: int = 4,
        drop_last: bool = True,
        pin_memory: bool = True,
        no_transform: bool = False,
        no_val_transform: bool = False,
        feature_subset: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()

        self.dataset_dir = Path(dataset_dir)
        self.sessions = sessions
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.transform_config = transform_config
        self.no_transform = no_transform
        self.no_val_transform = no_val_transform
        self.feature_subset = feature_subset

        self._train_examples: Optional[List[Dict]] = None
        self._val_examples: Optional[List[Dict]] = None

    def _gather_examples(self, split: str) -> List[Dict]:
        assert split in {"train", "val"}
        examples: List[Dict] = []
        h5_name = {"train": "data_train.hdf5", "val": "data_val.hdf5"}[split]
        for day_idx, sess in enumerate(self.sessions):
            h5_path = self.dataset_dir / sess / h5_name
            if not h5_path.exists():
                raise FileNotFoundError(str(h5_path))
            with h5py.File(str(h5_path), "r") as f:
                for key in f.keys():
                    if key.startswith("trial_"):
                        trial_idx = int(key.split("_")[1])
                        examples.append(
                            {
                                "session_path": str(h5_path),
                                "trial_idx": trial_idx,
                                "day_idx": day_idx,
                            }
                        )
        return examples

    def get_transform(
        self,
        mode: str = "train",
        jitter_range=[0.8, 1.0],
        jitter_max_start=200,
        channeldropout_prob=0.5,
        channeldropout_rate=0.2,
        scaleaugmnet_range=[0.95, 1.05],
        sample_window=32,
        transform_list=["jitter", "channeldrop", "scale"],
        no_jitter=False,
        cutdim=253,
        **kwargs,
    ):
        if self.no_transform:
            return None
        if mode == "train":
            transforms_ = []
            if "cutdim" in transform_list:
                transforms_.append(CutDim(cutdim))
            if "jitter" in transform_list and not no_jitter:
                transforms_.append(Jitter(jitter_range, jitter_max_start))
            if "channeldrop" in transform_list:
                transforms_.append(ChannelDrop(channeldropout_prob, channeldropout_rate))
            if "scale" in transform_list:
                transforms_.append(ScaleAugment(scaleaugmnet_range))
            if "sample_window" in transform_list:
                transforms_.append(SampleWindow(sample_window))
            return transforms.Compose(transforms_)
        else:
            return None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            if self._train_examples is None:
                self._train_examples = self._gather_examples("train")
            if self._val_examples is None:
                self._val_examples = self._gather_examples("val")

    def train_dataloader(self):
        dataset = RNNTMEADataset(
            self._train_examples,
            feature_subset=self.feature_subset,
            transform=self.get_transform("train", **self.transform_config),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=RNNTMEADataset.collate,
        )
        return loader

    def val_dataloader(self):
        transform = None if self.no_val_transform else self.get_transform("train", **self.transform_config)
        dataset = RNNTMEADataset(
            self._val_examples,
            feature_subset=self.feature_subset,
            transform=transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=RNNTMEADataset.collate,
        )
        return loader


class CutDim(object):
    def __init__(self, cutdim=253):
        self.cutdim = cutdim

    def __call__(self, x):
        return x[:, : self.cutdim]


class ChannelDrop(object):
    def __init__(self, apply_prob=0.5, dropout=0.2):
        self.apply_prob = apply_prob
        self.dropout = dropout
        try:
            self.is_range = len(dropout) > 1
        except Exception:
            self.is_range = False

    def __call__(self, x):
        if random.uniform(0, 1) < self.apply_prob:
            if self.is_range:
                dropout = np.random.uniform(self.dropout[0], self.dropout[1])
            else:
                dropout = self.dropout
            drop_mask = np.random.uniform(size=x.shape[1]) < dropout
            x[:, drop_mask] = 0
        return x


class Jitter(object):
    def __init__(self, fraction_range=[0.8, 1.0], max_start=100):
        self.max_start = 400
        self.fraction_pool = np.linspace(fraction_range[0], fraction_range[1], 5)

    def __call__(self, x):
        fraction = np.random.choice(self.fraction_pool, 1)[0]
        start_f = np.random.uniform() * (1 - fraction)
        end_f = start_f + fraction
        si, ei = int(len(x) * start_f), max(len(x), len(x) * end_f)
        si = min(self.max_start, si)
        x = x[si:ei]
        return x


class ScaleAugment(object):
    def __init__(self, range_):
        self.range = range_

    def __call__(self, x):
        scale = np.random.uniform(self.range[0], self.range[1], size=x.shape[1])
        x = x * scale[None, :]
        return x


class SampleWindow(object):
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, x):
        onset = np.random.uniform(0, len(x) - self.window_size)
        onset = min(len(x) - self.window_size, int(onset))
        x = x[onset : onset + self.window_size]
        return x
