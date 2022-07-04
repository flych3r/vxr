"""Encoder."""

from __future__ import annotations

from pytorch_lightning import LightningModule
from transformers import ViTModel


class VitEncoder(LightningModule):
    """Visual Extractor Encoder using ViT Base 16-224."""

    def __init__(
        self,
        pre_trained: str = 'google/vit-base-patch16-224-in21k',
        freeze: bool = False
    ):
        super().__init__()
        self.model = ViTModel.from_pretrained(pre_trained)
        self.config = self.model.config

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.forward = self.model.forward
