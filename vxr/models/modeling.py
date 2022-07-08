"""XrayReportGeneration model."""

from __future__ import annotations

from collections import OrderedDict, UserDict
from copy import deepcopy

import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, PreTrainedModel

from vxr.models.configuration import XrayReportGenerationConfig


class XrayReportGeneration(PreTrainedModel):
    """Model for generating x-ray reports from images."""
    config_class = XrayReportGenerationConfig
    base_model_prefix = 'xrrg'

    def __init__(self, config: XrayReportGenerationConfig):
        """Create a XrayReportGeneration from configuration.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.encoder: AutoModel = AutoModel.from_pretrained(
            self.config.pretrained_encoder.name_or_path
        )
        self.decoder: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.pretrained_decoder.name_or_path
        )
        self.main_input_name: str = self.encoder.main_input_name
        if self.config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        try:
            del self.decoder.encoder
        except AttributeError:
            pass

    def forward(
        self,
        pixel_values: torch.FloatTensor | UserDict = None,
        labels: torch.LongTensor = None,
        decoder_attention_mask: torch.LongTensor = None,
        encoder_outputs: OrderedDict = None,
        decoder_input_ids: torch.LongTensor = None,
        **kwargs
    ) -> torch.LongTensor | OrderedDict:
        """Forward pass on model.

        Args:
            pixel_values:
                Image pixel values
            labels:
                Labels for computing the sequence classification/regression loss.
            encoder_outputs:
                Hidden states at the output of the last layer of the encoder.
                Used in the cross-attention of the decoder
            decoder_input_ids:
                Indices of decoder input sequence tokens in the vocabulary.

        Returns:
            logits on training or output tokens on validation
        """
        if encoder_outputs is None:
            encoder_outputs = self.encoder(pixel_values)

        return self.decoder(
            encoder_outputs=encoder_outputs,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            decoder_input_ids=decoder_input_ids
        )

    def get_encoder(self) -> AutoModelForSeq2SeqLM:
        """Return a copy of the model encoder."""
        return deepcopy(self.encoder)

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, encoder_outputs: OrderedDict, **kwargs
    ) -> dict[str, torch.LongTensor | OrderedDict]:
        """Custom behavior to prepare inputs in the generate method."""
        return {
            'decoder_input_ids': input_ids,
            'encoder_outputs': encoder_outputs
        }
