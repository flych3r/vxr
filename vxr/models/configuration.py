"""XrayReportGeneration model configuration."""

from __future__ import annotations

import json

from transformers import AutoConfig, PretrainedConfig


class XrayReportGenerationConfig(PretrainedConfig):
    """This is the configuration class to store the configuration."""
    attribute_map = {
        'hidden_size': 'dim',
        'd_model': 'dim'
    }

    def __init__(
        self,
        pretrained_encoder: str | dict = 'google/vit-base-patch16-224-in21k',
        pretrained_decoder: str | dict = 'google/t5-efficient-base',
        use_pretrained_encoder: bool = True,
        use_pretrained_decoder: bool = True,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        **kwargs
    ):
        """Creates a new configuration.

        Args:
            pretrained_encoder: encoder model
            pretrained_decoder: decoder model
            use_pretrained_encoder: use only encoder architecture, no pretrained weights
            use_pretrained_decoder: use only decoder architecture, no pretrained weights
            freeze_encoder: freeze encoder weights for training
            freeze_decoder: freeze decoder weights for training

        Raises:
            ValueError:
                Encoder and decoder model sizes are not compatible.
        """
        super().__init__(**kwargs)

        self.pretrained_encoder = AutoConfig.from_pretrained(pretrained_encoder)
        self.pretrained_decoder = AutoConfig.from_pretrained(pretrained_decoder)
        enc_size = self.pretrained_encoder.hidden_size
        dec_size = self.pretrained_decoder.d_model
        if enc_size != dec_size:
            raise ValueError(
                f'Encoder and Decoder must have same sizes ({enc_size} != {dec_size})'
            )

        self.dim = enc_size
        self.architectures = (
            self.pretrained_encoder.architectures
            + self.pretrained_decoder.architectures
        )
        self.use_pretrained_encoder = use_pretrained_encoder
        self.use_pretrained_decoder = use_pretrained_decoder
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.is_encoder_decoder = True
        self.decoder_start_token_id = self.pretrained_decoder.pad_token_id

    def to_json_string(self, use_diff: bool = True) -> str:
        """Serializes this instance to a JSON string.

        Args:
            use_diff:
                If set to `True`, only the difference between the config
                instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`:
                String containing all the attributes that make up this
                configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        config_dict['pretrained_encoder'] = config_dict.pop(
            'pretrained_encoder'
        ).to_dict()
        config_dict['pretrained_decoder'] = config_dict.pop(
            'pretrained_decoder'
        ).to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + '\n'
