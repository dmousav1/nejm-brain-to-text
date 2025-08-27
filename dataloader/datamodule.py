import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class MEAHDF5Dataset(Dataset):
    """
    Map-style dataset that reads MEA feature trials from native HDF5 and returns
    RNNT-ready items (inputs, targets, and their lengths).
    Each item corresponds to a single trial.
    """

    def __init__(
        self,
        examples: List[Dict],
        feature_subset: Optional[List[int]] = None,
    ):
        super().__init__()
        self.examples = examples
        self.feature_subset = feature_subset

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[index]
        session_path: str = ex["session_path"]
        trial_idx: int = ex["trial_idx"]
        day_idx: int = ex["day_idx"]

        with h5py.File(session_path, "r") as f:
            g = f[f"trial_{trial_idx:04d}"]
            features = g["input_features"][:]
            if self.feature_subset is not None and len(self.feature_subset) > 0:
                features = features[:, self.feature_subset]
            targets = g["seq_class_ids"][:]

            # Optional metadata
            block_num = g.attrs.get("block_num", -1)
            trial_num = g.attrs.get("trial_num", -1)

        features_tensor = torch.from_numpy(features).float()  # [T, C]
        targets_tensor = torch.from_numpy(targets).long()  # [U]

        item = {
            "inputs": features_tensor,
            "targets": targets_tensor,
            "input_length": torch.tensor(features_tensor.shape[0], dtype=torch.long),
            "target_length": torch.tensor(targets_tensor.shape[0], dtype=torch.long),
            "day_index": torch.tensor(day_idx, dtype=torch.long),
            "block_num": torch.tensor(block_num, dtype=torch.long),
            "trial_num": torch.tensor(trial_num, dtype=torch.long),
        }
        return item

    @staticmethod
    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Inputs: list of [T_i, C]
        inputs = [b["inputs"] for b in batch]
        input_lengths = torch.tensor([x.shape[0] for x in inputs], dtype=torch.long)
        inputs_padded = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True, padding_value=0.0
        )  # [B, T_max, C]

        # Targets: list of [U_i]
        targets = [b["targets"] for b in batch]
        target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
        targets_padded = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=0
        )  # [B, U_max]

        out = {
            "inputs": inputs_padded,
            "input_lengths": input_lengths,
            "targets": targets_padded,
            "target_lengths": target_lengths,
            "day_indices": torch.stack([b["day_index"] for b in batch]),
            "block_nums": torch.stack([b["block_num"] for b in batch]),
            "trial_nums": torch.stack([b["trial_num"] for b in batch]),
        }
        return out


class MEAHDF5DataModule(pl.LightningDataModule):
    """
    Lightning DataModule that builds RNNT-ready DataLoaders from native MEA HDF5 files.

    Expects a directory structure like:
        dataset_dir/<session>/data_train.hdf5
        dataset_dir/<session>/data_val.hdf5
        (optional) dataset_dir/<session>/data_test.hdf5
    """

    def __init__(
        self,
        dataset_dir: str,
        sessions: List[str],
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        drop_last: bool = True,
        pin_memory: bool = True,
        feature_subset: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.sessions = sessions
        self.batch_size = batch_size
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.feature_subset = feature_subset

        self._train_examples: Optional[List[Dict]] = None
        self._val_examples: Optional[List[Dict]] = None
        self._test_examples: Optional[List[Dict]] = None

    def _gather_examples(self, split: str) -> List[Dict]:
        assert split in {"train", "val", "test"}
        examples: List[Dict] = []
        for day_idx, sess in enumerate(self.sessions):
            h5_name = {
                "train": "data_train.hdf5",
                "val": "data_val.hdf5",
                "test": "data_test.hdf5",
            }[split]
            h5_path = self.dataset_dir / sess / h5_name
            if not h5_path.exists():
                # Allow missing test split; skip silently
                if split == "test":
                    continue
                raise FileNotFoundError(f"Missing HDF5 file for {split}: {h5_path}")

            with h5py.File(str(h5_path), "r") as f:
                # trial_0000, trial_0001, ...
                for key in f.keys():
                    if re.match(r"^trial_\\d{4}$", key):
                        trial_idx = int(key.split("_")[1])
                        examples.append(
                            {
                                "session_path": str(h5_path),
                                "trial_idx": trial_idx,
                                "day_idx": day_idx,
                            }
                        )
        return examples

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self._train_examples is None:
                self._train_examples = self._gather_examples("train")
            if self._val_examples is None:
                self._val_examples = self._gather_examples("val")
        if stage in (None, "test"):
            if self._test_examples is None:
                self._test_examples = self._gather_examples("test")

    def train_dataloader(self) -> DataLoader:
        dataset = MEAHDF5Dataset(self._train_examples, feature_subset=self.feature_subset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=MEAHDF5Dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = MEAHDF5Dataset(self._val_examples, feature_subset=self.feature_subset)
        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=MEAHDF5Dataset.collate,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = MEAHDF5Dataset(self._test_examples or [], feature_subset=self.feature_subset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory,
            collate_fn=MEAHDF5Dataset.collate,
        )
