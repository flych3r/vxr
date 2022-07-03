"""Datamodule."""

from __future__ import annotations

from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import ImageFeatureExtractionMixin, PreTrainedTokenizer

from vxr.utils.data.dataset import XRayReportDataset


class XRayReportDataModule(LightningDataModule):
    """DataModule class that contains the X-ray images and reports."""

    def __init__(
        self,
        image_dir: Path,
        ann_path: Path,
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        transform: ImageFeatureExtractionMixin,
        batch_size: int = 32,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.ann_path = ann_path
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.transform = transform
        self.batch_size = batch_size
        self.dataset: dict[str, XRayReportDataset] = dict()

    def setup(self, stage: str = None):
        """Initialize the train, val and test datasets."""
        for split in ['train', 'val', 'test']:
            self.dataset[split] = XRayReportDataset(
                split,
                self.image_dir,
                self.ann_path,
                self.max_length,
                self.tokenizer,
                self.transform,
            )

    def train_dataloader(self) -> DataLoader[XRayReportDataset]:
        """Create the train dataloader."""
        return DataLoader(self.dataset['train'], batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader[XRayReportDataset]:
        """Create the val dataloader."""
        return DataLoader(self.dataset['val'], batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader[XRayReportDataset]:
        """Create the test dataloader."""
        return DataLoader(self.dataset['test'], batch_size=self.batch_size)
