"""Generation."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from vxr.models.encoder_decoder import XrayReportGeneration


def beam_search(
    model: XrayReportGeneration,
    encoder_outputs: torch.FloatTensor,
    beam_width: int = None,
    penalty: float = None
) -> torch.LongTensor:
    """
    BEAM searched report tokens.

    Parameters
    ----------
    model
        model to use for predictions
    encoder_outputs
        features extracted by the encoder
    beam width
        number of nodes of beam search
    penalty
        penalty factor for token repetitions


    Returns
    -------
    ret
        generated report token ids

    References
    ----------
    <https://github.com/jarobyte91/pytorch_beam_search/blob/master/src/pytorch_beam_search/seq2seq/search_algorithms.py>
    """
    beam_width = model.beam_width if beam_width is None else beam_width
    penalty = model.repetition_penalty if penalty is None else penalty
    batch_size = len(encoder_outputs.last_hidden_state)

    decoder_tokens = torch.full(
        (batch_size, 1),
        model.bos_token_id,
        dtype=torch.long,
        device=model.device,
    )

    outputs = model.decoder(
        encoder_outputs=encoder_outputs, decoder_input_ids=decoder_tokens
    )
    next_token_logits = outputs.logits[:, -1, :]
    vocabulary_size = next_token_logits.shape[-1]

    beam_probabilities, next_tokens = (
        next_token_logits.squeeze().log_softmax(-1).topk(k=beam_width, axis=-1)
    )

    next_tokens = next_tokens.reshape(-1, 1)
    decoder_tokens = decoder_tokens.repeat((beam_width, 1))
    decoder_tokens = torch.cat((decoder_tokens, next_tokens), axis=-1)

    for _ in range(model.max_length):
        beam_dataset = TensorDataset(
            encoder_outputs.last_hidden_state.repeat_interleave(beam_width, dim=0),
            decoder_tokens,
        )
        beam_next_token_logits = []
        for eo, dt in DataLoader(beam_dataset, batch_size=batch_size):
            beam_outputs = model.decoder(encoder_outputs=[eo], decoder_input_ids=dt)
            beam_next_token_logits.append(beam_outputs.logits[:, -1, :])

        beam_next_token_logits_penalized = repetition_penalty(
            torch.cat(beam_next_token_logits, axis=0), decoder_tokens, penalty
        )

        next_probabilities = beam_next_token_logits_penalized.log_softmax(-1)
        next_probabilities = next_probabilities.reshape(
            (-1, beam_width, next_probabilities.shape[-1])
        )
        beam_probabilities = beam_probabilities.unsqueeze(-1) + next_probabilities
        beam_probabilities = beam_probabilities.flatten(start_dim=1)

        beam_probabilities, next_tokens = beam_probabilities.topk(
            k=beam_width, dim=-1
        )
        best_candidates = (next_tokens / vocabulary_size).long()
        next_tokens = torch.remainder(next_tokens, vocabulary_size).reshape(-1, 1)
        best_candidates += torch \
            .arange(
                len(decoder_tokens) // beam_width,
                device=model.device
            ) \
            .unsqueeze(-1) \
            .multiply(beam_width)

        decoder_tokens = decoder_tokens[best_candidates].flatten(end_dim=-2)
        decoder_tokens = torch.cat((decoder_tokens, next_tokens), dim=1)

    decoder_tokens = decoder_tokens.reshape(
        (-1, beam_width, decoder_tokens.shape[-1])
    )
    best_beam = beam_probabilities.argmax(dim=1)[:, None, None]
    decoder_tokens = decoder_tokens.take_along_dim(best_beam, 1).squeeze(1)

    return decoder_tokens


def greedy_search(
    model: XrayReportGeneration,
    encoder_outputs: torch.FloatTensor,
    penalty: float = None
) -> torch.LongTensor:
    """
    Greedy searched report tokens.

    Parameters
    ----------
    model
        model to use for predictions
    encoder_outputs
        features extracted by the encoder
    penalty
        penalty factor for token repetitions

    Returns
    -------
    ret
        generated report token ids
    """
    penalty = model.repetition_penalty if penalty is None else penalty
    batch_size = len(encoder_outputs.last_hidden_state)

    decoder_tokens = torch.full(
        (batch_size, 1),
        model.bos_token_id,
        dtype=torch.long,
        device=model.device,
    )

    for _ in range(model.max_length):
        outputs = model.decoder(
            encoder_outputs=encoder_outputs, decoder_input_ids=decoder_tokens
        )
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = repetition_penalty(
            next_token_logits,
            decoder_tokens,
            penalty
        )

        next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
        if torch.eq(next_token_id[:, -1], model.eos_token_id).all():
            break
        decoder_tokens = torch.cat([decoder_tokens, next_token_id], dim=-1)

    return decoder_tokens


def repetition_penalty(
    logits: torch.FloatTensor,
    seen_tokens: torch.LongTensor,
    penalty: float
) -> torch.FloatTensor:
    """
    Penalize token repetitions on prediction.

    Parameters
    ----------
    logits
        output logits
    seen_tokens
        previously generated tokens
    penalty
        repetition penalty factor

    Returns
    -------
    ret
        logits token with repetitions penalized
    """
    logits_penalty = torch.gather(logits, 1, seen_tokens)
    logits_penalty = torch.where(
        logits_penalty < 0, logits_penalty * penalty, logits_penalty / penalty
    )
    logits.scatter_(1, seen_tokens, logits_penalty)
    return logits
