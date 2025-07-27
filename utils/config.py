# utils/config.py
import yaml
from dataclasses import dataclass
from typing import Any


@dataclass
class DataConfig:
    topo_path: str
    alarm_path: str
    batch_size: int
    max_len: int
    window_milliseconds: float
    step_milliseconds: float
    num_train: int
    num_val: int
    num_test: int


@dataclass
class GNNConfig:
    hidden_channels: int
    num_layers: int
    dropout: float


@dataclass
class TransformerConfig:
    emb_dim: int
    nhead: int
    hid_dim: int
    nlayers: int
    max_len: int
    dropout: float


@dataclass
class TrainingConfig:
    epochs: int
    lr: float
    weight_decay: float
    lr_step_size: int
    early_stop_patience: int
    focal_alpha : float
    focal_gamma : float


@dataclass
class Config:
    data: DataConfig
    gnn: GNNConfig
    transformer: TransformerConfig
    training: TrainingConfig


def load_config(path: str = "configs/config.yaml") -> Config:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return Config(
        data=DataConfig(**cfg['data']),
        gnn=GNNConfig(**cfg['model']['gnn']),
        transformer=TransformerConfig(**cfg['model']['transformer']),
        training=TrainingConfig(
            epochs=int(cfg['training']['epochs']),
            lr=float(cfg['training']['lr']),
            weight_decay=float(cfg['training']['weight_decay']),
            lr_step_size=int(cfg['training']['lr_step_size']),
            early_stop_patience=int(cfg['training']['early_stop_patience']),
            focal_alpha=float(cfg['training']['focal_alpha']),
            focal_gamma=float(cfg['training']['focal_gamma']),
        ),
    )
