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
        encoder_config: str | PretrainedConfig = 'google/vit-base-patch16-224-in21k',
        decoder_config: str | PretrainedConfig = 'google/t5-efficient-base',
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

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.use_pretrained_encoder = False
        self.use_pretrained_decoder = False

        if isinstance(self.encoder_config, str):
            self.encoder_config = AutoConfig.from_pretrained(self.encoder_config)
            self.use_pretrained_encoder = use_pretrained_encoder

        if isinstance(self.decoder_config, str):
            self.decoder_config = AutoConfig.from_pretrained(self.decoder_config)
            self.use_pretrained_decoder = use_pretrained_decoder

        if isinstance(self.encoder_config, dict):
            encoder_type = self.encoder_config['model_type']
            encoder_type_config = AutoConfig.for_model(encoder_type)
            self.encoder_config = encoder_type_config.from_dict(self.encoder_config)

        if isinstance(self.decoder_config, dict):
            decoder_type = self.decoder_config['model_type']
            decoder_type_config = AutoConfig.for_model(decoder_type)
            self.decoder_config = decoder_type_config.from_dict(self.decoder_config)

        enc_size = (
            self.encoder_config.hidden_size
            if hasattr(self.encoder_config, 'hidden_size')
            else self.encoder_config.hidden_sizes[-1]
        )
        dec_size = self.decoder_config.d_model
        if enc_size != dec_size:
            raise ValueError(
                f'Encoder and Decoder must have same sizes ({enc_size} != {dec_size})'
            )

        self.dim = enc_size
        self.architectures = (
            self.encoder_config.architectures
            + self.decoder_config.architectures
        )
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder = freeze_decoder
        self.is_encoder_decoder = True
        self.decoder_start_token_id = self.decoder_config.pad_token_id

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

        encoder_config = config_dict.pop('encoder_config')
        decoder_config = config_dict.pop('decoder_config')

        if not isinstance(encoder_config, dict):
            encoder_config = encoder_config.to_dict()
        if not isinstance(decoder_config, dict):
            decoder_config = decoder_config.to_dict()

        config_dict['encoder_config'] = encoder_config
        config_dict['decoder_config'] = decoder_config

        return json.dumps(config_dict, indent=2, sort_keys=True) + '\n'
