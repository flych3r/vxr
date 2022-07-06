"""Dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import ImageFeatureExtractionMixin, PreTrainedTokenizer


class XrayReportDataset(Dataset):
    """Dataset class that contains the X-ray images and reports."""

    def __init__(
        self,
        split: str,
        image_dir: Path,
        ann_path: Path,
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        transform: ImageFeatureExtractionMixin,
    ):
        self.split = split
        self.image_dir = image_dir
        self.ann_path = ann_path
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.transform = transform
        with open(self.ann_path, 'r') as f:
            self.annotations = json.load(f)

        self.data = {
            idx: {
                **ann,
                **tokenizer(
                    ann['report'],
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                ),
            }
            for idx, ann in enumerate(self.annotations[self.split])
        }

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> tuple[str, torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        """
        Retrieve an item from the dataset.

        Parameters
        ----------
        index
            dataset index

        Returns
        -------
        ret
            id of the image, transformed image, tokenized report, report attention mask
        """
        item = self.data[index]
        image_id: str = item['id']
        image_path = item['image_path']
        image = Image.open(self.image_dir / image_path[0]).convert('RGB')
        image_transformed: torch.FloatTensor = self.transform(  # type: ignore[operator]
            image, return_tensors='pt'
        ).pixel_values[0]
        input_ids = torch.LongTensor(item['input_ids'])
        attention_mask = torch.LongTensor(item['attention_mask'])
        return image_id, image_transformed, input_ids, attention_mask
