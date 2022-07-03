"""Reporter."""

import evaluate
import torch
from pytorch_lightning import LightningModule

from vxr.models.visual_extractor import ResnetVisualExtractor, VitVisualExtractor


class XrayReportGeneration(LightningModule):
    """Model for generating x-ray reports from images."""

    def __init__(self, visual_extractor: str):
        super(XrayReportGeneration, self).__init__()
        if visual_extractor == 'resnet':
            self.visual_extractor = ResnetVisualExtractor()
        elif visual_extractor == 'vit':
            self.visual_extractor = VitVisualExtractor()
        else:
            raise ValueError('visual-extractor must be resnet or vit')

        self.metrics = {
            'BLEU-1': evaluate.load('bleu'),
            'BLEU-2': evaluate.load('bleu'),
            'BLEU-3': evaluate.load('bleu'),
            'BLEU-4': evaluate.load('bleu'),
            'METEOR': evaluate.load('meteor'),
        }

    def forward(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Forward pass for report generator.

        Parameters
        ----------
        batch
            batch with images ids, images, tokens and masks

        Returns
        -------
        ret
        """
        ids, imgs, tokens, masks = batch
        att_feats, fc_feats = self.visual_extractor(imgs)
        return att_feats, fc_feats
