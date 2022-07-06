"""Decoder."""

from __future__ import annotations

import torch
from pytorch_lightning import LightningModule
from transformers import AutoModelForSeq2SeqLM


class LanguageModelDecoder(LightningModule):
    """Language Model Decoder."""

    def __init__(self, pre_trained: str = 'google/t5-efficient-base'):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pre_trained)
        try:
            del self.model.encoder
        except AttributeError:
            pass
        self.config = self.model.config

    def forward(
        self,
        encoder_outputs: torch.FloatTensor,
        labels: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
    ):
        """
        Forward pass fot the decoder model.

        Parameters
        ----------
        encoder_outputs
            features extracted by the encoder
        labels
            sentence tokens
        decoder_input_ids
            generation tokens

        Returns
        -------
        ret
            logits on training or output tokens on validation
        """
        if self.training:
            return self.model(encoder_outputs=encoder_outputs, labels=labels)
        return self.model(
            encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids
        )
