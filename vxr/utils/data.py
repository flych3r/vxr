"""Dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoFeatureExtractor, AutoTokenizer


class XrayReportDataset:
    """Class for creating datasets."""

    def __init__(
        self,
        image_dir: Path,
        ann_path: Path,
        max_length: int,
        tokenizer: AutoTokenizer,
        transforms: AutoFeatureExtractor
    ):
        """Create train, validation and test datasets.

        Args:
            image_dir: path to directory of images
            ann_path: path to annotations json
            max_length: maximum size of the tokenizer ids output
            tokenizer: text tokenizer
            transforms: image transformations
        """
        super().__init__()
        self.image_dir = image_dir
        self.ann_path = ann_path
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.transforms = transforms
        self._setup()

    def _setup(self):
        """Creates the datasets."""
        with open(self.ann_path) as f:
            annotations = json.load(f)

        data = defaultdict(lambda: defaultdict(list))
        for split in ['train', 'val', 'test']:
            for sample in annotations[split]:
                data[split]['image_id'].append(sample['id'])
                data[split]['image_path'].append(sample['image_path'][0])
                data[split]['report'].append(sample['report'])

        datasets = {
            s: Dataset.from_dict(d).map(self.process_batch, batched=True)
            for s, d in data.items()
        }
        self.train = datasets['train']
        self.validation = datasets['val']
        self.test = datasets['test']

    def process_batch(self, batch: dict) -> dict:
        """Process an input batch generating pixel_values and input_ids.

        Args:
            batch:
                dataset batch

        Returns:
            batch with transformed image pixel_values, report input_ids and attn_mask
        """
        images = [
            Image.open(self.image_dir / img).convert('RGB')
            for img in batch['image_path']
        ]
        image_transformed = self.transforms(images)
        tokens = self.tokenizer(
            batch['report'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
        )
        return {
            **batch,
            **image_transformed,
            **tokens
        }

    def __repr__(self) -> str:
        """String representation."""
        data_dict = {
            'train': {
                'features': self.train.column_names,
                'num_rows': self.train.num_rows,
            },
            'validation': {
                'features': self.validation.column_names,
                'num_rows': self.validation.num_rows,
            },
            'test': {
                'features': self.test.column_names,
                'num_rows': self.test.num_rows,
            },

        }
        return json.dumps(data_dict, indent=2, sort_keys=False) + '\n'


def collate_fn(batch: dict) -> dict:
    """Collate function from data to model.

    Args:
        batch:
            dataset batch

    Returns:
        pixel_values, labels and decoder_attention_mask
    """
    return {
        'pixel_values': torch.stack([torch.tensor(x['pixel_values']) for x in batch]),
        'labels': torch.tensor([x['input_ids'] for x in batch]),
        'decoder_attention_mask': torch.tensor([x['attention_mask'] for x in batch]),
    }
