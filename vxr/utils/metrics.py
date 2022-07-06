"""Metrics."""

from __future__ import annotations

from typing import Callable

import evaluate


def nlp_metrics() -> dict[str, Callable]:
    """Create metrics for BLEU-{1, 4} and Meteor."""
    return {
        'BLEU': evaluate.load('bleu').compute,
        'METEOR': evaluate.load('meteor').compute
    }


def eval_generated_texts(
    metrics: dict[str, Callable],
    references: list[str],
    predictions: list[str],
    stage: str
) -> dict[str, float]:
    """Evaluate generated texts using metrics."""
    bleu_score = metrics['BLEU'](references=references, predictions=predictions)
    meteor_score = metrics['METEOR'](references=references, predictions=predictions)

    return {
        **{
            f'{stage}/BLEU-{i}': s
            for i, s in enumerate(bleu_score['precisions'], start=1)
        },
        f'{stage}/METEOR': meteor_score['meteor']
    }
