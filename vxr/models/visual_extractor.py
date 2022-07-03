"""Visual Extractor."""

from __future__ import annotations

from collections import OrderedDict

import timm
import torch


class BaseVisualExtractor(torch.nn.Module):
    """Base visual extractor pulling pre-trained models from timm checkpoint."""

    def __init__(self, pre_trained: str, freeze: bool = False) -> None:
        super().__init__()
        pretrained_model = timm.create_model(pre_trained, pretrained=True)
        modules = list(pretrained_model.named_children())[:-2]
        self.model = torch.nn.Sequential(OrderedDict(modules))
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False


class ResnetVisualExtractor(BaseVisualExtractor):
    """Visual Extractor using ResNet101."""

    def __init__(self, freeze: bool = False):
        super().__init__('resnet101', freeze)

    def forward(
        self, images: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Forward pass for visual feature extraction.

        Parameters
        ----------
        images
            batch of images

        Returns
        -------
        ret
            resnet features (batch, 49, 2048) and averages (batch, 2048)
        """
        patch_feats = self.model(images)
        avg_feats = patch_feats.mean(dim=(2, 3))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats


class VitVisualExtractor(BaseVisualExtractor):
    """Visual Extractor using ViT Base 16-224."""

    def __init__(self, freeze: bool = False):
        super().__init__('vit_base_patch16_224_in21k', freeze)

    def forward(
        self, images: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Forward pass for visual feature extraction.

        Parameters
        ----------
        images
            batch of images

        Returns
        -------
        ret
            vit features (batch, 196, 768) and averages (batch, 768)
        """
        patch_feats = self.model(images)
        avg_feats = patch_feats.mean(dim=1)
        return patch_feats, avg_feats
