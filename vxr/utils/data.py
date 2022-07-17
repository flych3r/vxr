"""Dataset."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, AutoTokenizer


class XrayReportDataset(Dataset):
    """Dataset class that contains the X-ray images and reports."""

    def __init__(
        self,
        split: str,
        image_dir: Path,
        ann_path: Path,
        max_length: int,
        tokenizer: AutoTokenizer,
        transforms: AutoFeatureExtractor,
        sample: float = 1.0,
        seed: int = None
    ):
        """Create dataset.

        Args:
            split: data split (train, val, test)
            image_dir: path to directory of images
            ann_path: path to annotations json
            max_length: maximum size of the tokenizer ids output
            tokenizer: text tokenizer
            transforms: image transformations
            sample: amount of reduced data to sample
            seed: seed for sampling
        """
        super().__init__()

        self.split = split
        self.image_dir = image_dir
        self.ann_path = ann_path
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.sample = sample

        with open(self.ann_path, 'r') as f:
            annotations = json.load(f)

        self.data = annotations[split]
        if 0.0 < self.sample < 1.0:
            random = Random(seed)
            total = max(int(len(self.data) * self.sample), 1)
            self.data = random.sample(self.data, total)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(
        self, index: int
    ):
        """Retrieve an item from the dataset.

        Args:
            index: dataset index

        Returns:
            transformed image, tokenized report, report attention mask
        """
        item = self.data[index]
        image = Image.open(self.image_dir / item['image_path'][0]).convert('RGB')
        image_transformed = self.transforms(image, return_tensors='pt')

        tokens = self.tokenizer(
            item['report'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
        )

        return {
            'pixel_values': image_transformed['pixel_values'][0],
            'input_ids': torch.LongTensor(tokens['input_ids']),
            'attention_mask': torch.LongTensor(tokens['attention_mask']),
        }


class XrayReportData:
    """DataModule class that contains the X-ray images and reports."""

    def __init__(
        self,
        image_dir: Path,
        ann_path: Path,
        max_length: int,
        tokenizer: AutoTokenizer,
        transforms: AutoFeatureExtractor,
        sample: float = 1.0,
        seed: int = None
    ):
        """Create train, validation and test datasets.

        Args:
            image_dir: path to directory of images
            ann_path: path to annotations json
            max_length: maximum size of the tokenizer ids output
            tokenizer: text tokenizer
            transforms: image transformations
            sample: amount of reduced data to sample
            seed: seed for sampling
        """
        super().__init__()
        self.image_dir = image_dir
        self.ann_path = ann_path
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.sample = sample
        self.seed = seed
        data = self._setup()
        self.train = data['train']
        self.validation = data['val']
        self.test = data['test']

    def _setup(self):
        """Initialize the train, val and test datasets."""
        return {
            split: XrayReportDataset(
                split,
                self.image_dir,
                self.ann_path,
                self.max_length,
                self.tokenizer,
                self.transforms,
                self.sample,
                self.seed
            ) for split in ['train', 'val', 'test']
        }


def collate_fn(batch: dict) -> dict:
    """Collate function from data to model.

    Args:
        batch:
            dataset batch

    Returns:
        pixel_values, labels and decoder_attention_mask
    """
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['input_ids'] for x in batch]),
        'decoder_attention_mask': torch.stack([x['attention_mask'] for x in batch]),
    }
