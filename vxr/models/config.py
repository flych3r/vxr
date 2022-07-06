"""Config."""

from __future__ import annotations

from transformers import AutoConfig


def check_encoder_decoder_compatible(encoder: str, decoder: str):
    """
    Check the encoder and decoder model sizes.

    Parameters
    ----------
    encoder
        encoder model name or path
    decoder
        decoder model name or path

    Raises
    ------
    ValueError
        If model sizes are different
    """
    encoder_config = AutoConfig.from_pretrained(encoder)
    decoder_config = AutoConfig.from_pretrained(decoder)

    enc_size = encoder_config.hidden_size
    dec_size = decoder_config.d_model
    if enc_size != dec_size:
        raise ValueError(
            f'Encoder and Decoder must have same sizes ({enc_size} != {dec_size})'
        )
