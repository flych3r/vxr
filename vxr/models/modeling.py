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

        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()
        self.main_input_name: str = self.encoder.main_input_name
        self._reorder_cache = self.decoder._reorder_cache

    def _create_encoder(self) -> AutoModel:
        """Creates model encoder using config."""
        if self.config.use_pretrained_encoder:
            encoder: AutoModel = AutoModel.from_pretrained(  # type: ignore[no-redef]  # noqa: E501
                self.config.pretrained_encoder.name_or_path
            )
        else:
            encoder: AutoModel = AutoModel.from_config(  # type: ignore[no-redef]  # noqa: E501
                self.config.pretrained_encoder
            )

        if self.config.freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
        if hasattr(encoder, 'decoder'):
            del encoder.decoder
        return encoder

    def _create_decoder(self) -> AutoModelForSeq2SeqLM:
        """Creates model decoder using config."""
        if self.config.use_pretrained_decoder:
            decoder: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_pretrained(  # type: ignore[no-redef]  # noqa: E501
                self.config.pretrained_decoder.name_or_path
            )
        else:
            decoder: AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM.from_config(  # type: ignore[no-redef]  # noqa: E501
                self.config.pretrained_decoder
            )
        if self.config.freeze_decoder:
            for param in decoder.parameters():
                param.requires_grad = False
        if hasattr(decoder, 'encoder'):
            del decoder.encoder
        elif hasattr(decoder.model, 'encoder'):
            del decoder.model.encoder
        return decoder

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
            encoder_outputs = self.prepare_encoder_outputs(self.encoder(pixel_values))

        return self.decoder(
            encoder_outputs=encoder_outputs,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            decoder_input_ids=decoder_input_ids
        )

    def get_encoder(self) -> torch.nn.Module:
        """Return a copy of the model encoder."""
        class DummyEncoder(torch.nn.Module):
            def __init__(self, model) -> None:
                super().__init__()
                self.model = deepcopy(model)

            def forward(self, **kwargs):
                outputs = self.model.forward(
                    pixel_values=kwargs['pixel_values'],
                    output_hidden_states=kwargs['output_hidden_states'],
                    return_dict=kwargs['return_dict'],
                )
                return XrayReportGeneration.prepare_encoder_outputs(outputs)

        return DummyEncoder(self.encoder)

    @staticmethod
    def prepare_inputs_for_generation(
        input_ids: torch.LongTensor, encoder_outputs: OrderedDict, **kwargs
    ) -> dict[str, torch.LongTensor | OrderedDict]:
        """Custom behavior to prepare inputs in the generate method."""
        return {
            'decoder_input_ids': input_ids,
            'encoder_outputs': encoder_outputs
        }

    @staticmethod
    def prepare_encoder_outputs(encoder_outputs: OrderedDict) -> OrderedDict:
        """Custom behavior to prepare encoder outputs for decoder."""
        if len(encoder_outputs['last_hidden_state'].size()) == 4:
            encoder_outputs['last_hidden_state'] = encoder_outputs['last_hidden_state'] \
                .permute(0, 2, 3, 1) \
                .flatten(start_dim=1, end_dim=2)
        return encoder_outputs
