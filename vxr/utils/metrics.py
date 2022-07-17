"""Metrics."""

from __future__ import annotations

from typing import Callable

import evaluate
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from transformers import Trainer


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


def overfit_on_batch(
    trainer: Trainer,
    compute_metrics: Callable,
    max_length: int,
    epochs: int = 100,
    plot_curve: bool = True
) -> tuple[dict[str, float], torch.LongTensor, torch.LongTensor]:
    """Overfit trainer on a batch.

    Must recreate trainer and model after overfitting.

    Args:
        trainer: model trainer
        compute_metrics: metrics to measure overfit
        max_length: max length of generation
        epochs: number of epochs to overfit
        plot_curve: wether to plot the overfitting loss curve

    Returns:
        metrics, predictions and ground truths
    """
    device = trainer.model.device
    for batch in trainer.get_train_dataloader():
        batch = {k: v.to(device) for k, v in batch.items()}
        break

    trainer.create_optimizer()

    losses = []
    for _ in tqdm(range(epochs)):
        outputs = trainer.model(**batch)
        loss = outputs.loss
        loss.backward()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        losses.append(loss.detach().cpu().numpy())

    if plot_curve:
        plt.plot(losses)
        plt.ylabel('loss')
        plt.title('Overfit on batch loss curve')
        plt.show()

    labels = batch['labels'].cpu()

    with torch.no_grad():
        preds = trainer.model \
            .generate(batch['pixel_values'], max_length=max_length) \
            .cpu()

    return compute_metrics((preds, labels)), preds, labels
