import argparse
import os
import re
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as pl
import json
from lightning.pytorch.callbacks import ModelCheckpoint

# RNNT model components
from model.ecog2speech_rnnt import SpeechModel
from dataloader.datamodule import MEAHDF5Dataset

from omegaconf import OmegaConf
# from model_training.rnn_trainer import BrainToTextDecoder_Trainer


def _gather_examples(dataset_dir, sessions, split):
    assert split in {"train", "val"}
    examples = []
    for day_idx, sess in enumerate(sessions):
        h5_name = {"train": "data_train.hdf5", "val": "data_val.hdf5"}[split]
        h5_path = os.path.join(dataset_dir, sess, h5_name)
        if not os.path.exists(h5_path):
            if split == 'val':
                continue
            raise FileNotFoundError(f"Missing HDF5 file for {split}: {h5_path}")
        with h5py.File(h5_path, 'r') as f:
            for key in f.keys():
                if re.match(r"^trial_\d{4}$", key):
                    trial_idx = int(key.split('_')[1])
                    examples.append({
                        'session_path': h5_path,
                        'trial_idx': trial_idx,
                        'day_idx': day_idx,
                    })
    return examples


def _compute_max_target_id(examples):
    max_id = 0
    for ex in examples:
        with h5py.File(ex['session_path'], 'r') as f:
            g = f[f"trial_{ex['trial_idx']:04d}"]
            if 'seq_class_ids' in g:
                arr = g['seq_class_ids'][:]
                if arr.size > 0:
                    max_id = max(max_id, int(np.max(arr)))
    return int(max_id)


def rnnt_collate_from_mea(batch):
    base = MEAHDF5Dataset.collate(batch)
    # Convert targets to space-separated integers (strings) for tokenizer
    targets = base['targets']  # [B, U_max]
    lengths = base['target_lengths']  # [B]
    unit_texts = []
    for i in range(targets.shape[0]):
        l = int(lengths[i].item())
        ids = targets[i, :l].tolist()
        unit_texts.append(' '.join(str(int(x)) for x in ids))
    return {
        'ecogs': base['inputs'],
        'ecog_lens': base['input_lengths'],
        'texts': {'unit': unit_texts},
        # keep extra metadata if needed downstream
        'block_nums': base.get('block_nums'),
        'trial_nums': base.get('trial_nums'),
    }


def _list_h5_files(data_root, sessions, split_name):
    files = []
    for s in sessions:
        fp = os.path.join(data_root, s, f'data_{split_name}.hdf5')
        if os.path.exists(fp):
            files.append(fp)
    return files


def train_rnnt_from_hdf5(args):
    # Build train/val datasets from HDF5 using MEA dataset schema
    data_root = args['dataset']['dataset_dir']
    sessions = args['dataset']['sessions']
    train_examples = _gather_examples(data_root, sessions, 'train')
    val_examples = _gather_examples(data_root, sessions, 'val')

    if len(train_examples) == 0:
        raise RuntimeError(f'No train HDF5 trials found under {data_root}.')

    # Where to save artifacts (splits + checkpoints)
    output_dir = args.get('output_dir', 'model_training/trained_models/rnnt')
    os.makedirs(output_dir, exist_ok=True)

    # Persist the dataset splits for quick DataLoader reconstruction
    with open(os.path.join(output_dir, 'train_examples.json'), 'w') as f:
        json.dump(train_examples, f)
    with open(os.path.join(output_dir, 'val_examples.json'), 'w') as f:
        json.dump(val_examples, f)

    # Compute vocab size from label IDs for tokenizer/predictor alignment
    max_id = _compute_max_target_id(train_examples)
    km_n = max_id + 1  # tokenizer.pad_id = km_n, tokenizer.blank = km_n+1
    num_symbols = km_n + 2  # predictor vocab = pad + blank + ids

    batch_size = int(args['dataset'].get('batch_size', 32)) # Chooses 64 (in rnnt_args.yaml file) but if not found, defaults to 32
    num_workers = int(args['dataset'].get('num_dataloader_workers', 4))
    train_ds = MEAHDF5Dataset(train_examples)
    val_ds = MEAHDF5Dataset(val_examples) if len(val_examples) > 0 else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=rnnt_collate_from_mea, drop_last=True)
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=rnnt_collate_from_mea, drop_last=False)

    # RNNT model configs (simple, single-module bundle)
    bundle_configs = {
        'unit': {
            'target': 'unit',
            'tokenizer_type': 'HB',
            'tokenizer_configs': {
                'pre_tokenized': True,
                'km_n': km_n,
                'collapse': True
            },
            'transcriber_configs': {
                'MODEL': 'MEATranscriber',
                'input_dim': int(args['model']['n_input_features']),
                'unfold_window_size': 6,
                'unfold_stride': 2,
                'gaussian_sigma': 1.0,
                'proj_dim': 512,
                'dropout': 0.2,
                'rnn_hidden_dim': 512,
                'rnn_num_layers': 3,
                'bidirectional': False,
                'cnn_kernel_size': 3,
                'cnn_stride': 2,
                'post_linear_dim': 512,
                'output_dim': 1024
            },
            'predictor_configs': {
                'MODEL': 'EmbPredictor',
                'num_symbols': num_symbols,
                'output_dim': 512
            },
            'joiner_configs': {
                'MODEL': 'PremapJoinerv2',
                'transcriber_dim': 1024,
                'input_dim': 512,
                'output_dim': num_symbols,
                'activation': 'tanh',
                'hidden_dim': 512,
                'fix_linear': False
            }
        }
    }

    # Identity feature extractor; RNNT transcriber handles feature processing
    feature_extractor_configs = {
        'MODEL': 'DummyTranscriber',
        'input_dim': int(args['model']['n_input_features'])
    }

    loss_coef_instructions = {
        'unit_rnnt_loss': {'value': 1.0}
    }

    model = SpeechModel(
        loss_coef_instructions=loss_coef_instructions,
        non_val_loss=[],
        lr=float(args.get('lr_max', 0.001)),
        use_cosine_lr=True,
        T_max=int(args.get('lr_decay_steps', 100000)),
        skip_pred=False,
        feature_extractor_configs=feature_extractor_configs,
        bundle_configs=bundle_configs
    )

    # Trainer setup
    devices = None
    if torch.cuda.is_available():
        # Use a single GPU slot specified by config if present
        gpu_number = args.get('gpu_number', '0')
        try:
            gpu_index = int(gpu_number)
            devices = [gpu_index]
        except Exception:
            devices = 1

    # Per-epoch checkpointing
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='epoch{epoch:03d}-step{step:06d}',
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
        monitor=None,
    )

    max_epochs = 10 # fallback; RNNT often uses many steps; control via batches per epoch
    trainer = pl.Trainer(
        accelerator='gpu' if devices is not None else 'cpu',
        devices=devices,
        max_epochs=max_epochs,
        log_every_n_steps=50,
        gradient_clip_val=float(args.get('grad_norm_clip_value', 1.0)),
        default_root_dir=output_dir,
        callbacks=[ckpt_cb],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_rnnt', action='store_true', help='Train RNNT instead of baseline GRU-CTC')
    cli_args, _ = parser.parse_known_args()

    args = OmegaConf.load('model_training/rnnt_args.yaml')

    train_rnnt_from_hdf5(args)

    # if cli_args.use_rnnt:
    #     train_rnnt_from_hdf5(args)
    # else:
    #     trainer = BrainToTextDecoder_Trainer(args)
    #     metrics = trainer.train()