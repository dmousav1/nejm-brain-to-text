import argparse
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as pl

# RNNT model components
from model.ecog2speech_rnnt import SpeechModel
from model.tokenizer import GraphemeTokenizer

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer


class HDF5RNNTDataset(Dataset):
    """
    Minimal dataset that reads trials from Brain-To-Text HDF5 files and
    returns samples in the format expected by the RNNT pipeline:
      - ecog: FloatTensor [T, C]
      - ecog_len: int
      - text: dict with key 'phoneme' holding a plain string transcription
    """
    def __init__(self, file_paths):
        super().__init__()
        self.index = []  # list of (file_path, trial_key)
        for fp in file_paths:
            if not os.path.exists(fp):
                continue
            try:
                with h5py.File(fp, 'r') as f:
                    for key in f.keys():
                        g = f[key]
                        # require neural features and either transcription or seq_class_ids
                        if 'input_features' not in g:
                            continue
                        if 'transcription' in g or 'seq_class_ids' in g:
                            self.index.append((fp, key))
            except Exception:
                continue

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fp, key = self.index[i]
        with h5py.File(fp, 'r') as f:
            g = f[key]
            feats = g['input_features'][:].astype(np.float32)
            # Prefer character-level transcription when available
            if 'transcription' in g:
                trans = g['transcription'][:]
                if isinstance(trans, (bytes, bytearray, np.bytes_)):
                    text = trans.decode('utf-8', errors='ignore')
                else:
                    try:
                        text = ''.join([t.decode('utf-8', errors='ignore') if isinstance(t, (bytes, bytearray, np.bytes_)) else str(t) for t in trans])
                    except Exception:
                        text = str(trans)
            else:
                # Fallback: integer seq_class_ids -> join as space-separated numbers to form a stable string
                ids = g['seq_class_ids'][:]
                text = ' '.join([str(int(x)) for x in ids])

        return {
            'ecog': torch.from_numpy(feats),
            'ecog_len': feats.shape[0],
            'text': {
                # The RNNT bundle here is named 'phoneme' and expects this key
                'phoneme': text
            }
        }

    @staticmethod
    def collate(batch):
        ecogs = torch.nn.utils.rnn.pad_sequence([b['ecog'] for b in batch], batch_first=True, padding_value=0.0)
        ecog_lens = torch.tensor([b['ecog_len'] for b in batch], dtype=torch.long)
        texts = {'phoneme': [b['text']['phoneme'] for b in batch]}
        return {'ecogs': ecogs, 'ecog_lens': ecog_lens, 'texts': texts}


def _list_h5_files(data_root, sessions, split_name):
    files = []
    for s in sessions:
        fp = os.path.join(data_root, s, f'data_{split_name}.hdf5')
        if os.path.exists(fp):
            files.append(fp)
    return files


def train_rnnt_from_hdf5(args):
    # Build train/val datasets from HDF5
    data_root = args['dataset']['dataset_dir']
    sessions = args['dataset']['sessions']
    train_files = _list_h5_files(data_root, sessions, 'train')
    val_files = _list_h5_files(data_root, sessions, 'val')

    if len(train_files) == 0:
        raise RuntimeError(f'No train HDF5 files found under {data_root}.')
    if len(val_files) == 0:
        print('Warning: No val HDF5 files found. Validation will be skipped.')

    train_ds = HDF5RNNTDataset(train_files)
    val_ds = HDF5RNNTDataset(val_files) if len(val_files) > 0 else None

    batch_size = int(args['dataset'].get('batch_size', 32))
    num_workers = int(args['dataset'].get('num_dataloader_workers', 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=HDF5RNNTDataset.collate, drop_last=True)
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=HDF5RNNTDataset.collate, drop_last=False)

    # Derive vocabulary size from Grapheme tokenizer (no external g2p dependency)
    tmp_tok = GraphemeTokenizer(include_space=True)
    num_symbols = int(tmp_tok.blank) + 1  # final index is blank, symbols are 0..blank

    # RNNT model configs (simple, single-module bundle)
    bundle_configs = {
        'phoneme': {
            'tokenizer_type': 'GR',
            'tokenizer_configs': {
                'include_space': True
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

    # Top-level feature "extractor" is identity to pass raw ECoG to transcriber
    feature_extractor_configs = {
        'MODEL': 'MEATranscriber',
        'input_dim': int(args['model']['n_input_features'])
    }

    loss_coef_instructions = {
        'phoneme_rnnt_loss': {
            'value': 1.0
        }
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

    max_epochs = 1  # fallback; RNNT often uses many steps; control via batches per epoch
    trainer = pl.Trainer(
        accelerator='gpu' if devices is not None else 'cpu',
        devices=devices,
        max_epochs=max_epochs,
        log_every_n_steps=50,
        gradient_clip_val=float(args.get('grad_norm_clip_value', 1.0)),
        default_root_dir=args.get('output_dir', 'trained_models/rnnt')
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_rnnt', action='store_true', help='Train RNNT instead of baseline GRU-CTC')
    cli_args, _ = parser.parse_known_args()

    args = OmegaConf.load('rnn_args.yaml')

    train_rnnt_from_hdf5(args)

    # if cli_args.use_rnnt:
    #     train_rnnt_from_hdf5(args)
    # else:
    #     trainer = BrainToTextDecoder_Trainer(args)
    #     metrics = trainer.train()