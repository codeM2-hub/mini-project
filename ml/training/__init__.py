"""Training subpackage — Dataset, Trainer, and evaluation utilities."""

from .dataset import AudioActionDataset
from .trainer import Trainer

__all__ = ["AudioActionDataset", "Trainer"]
