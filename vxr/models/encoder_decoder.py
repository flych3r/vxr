"""EncoderDecoder."""

from __future__ import annotations

from typing import Callable

import torch
from pytorch_lightning import LightningModule
from transformers import PreTrainedTokenizer

from vxr.models.config import check_encoder_decoder_compatible
from vxr.models.decoder import LanguageModelDecoder
from vxr.models.encoder import VisualExtractorEncoder
from vxr.utils.generation import beam_search, greedy_search
from vxr.utils.metrics import eval_generated_texts


class XrayReportGeneration(LightningModule):
    """Model for generating x-ray reports from images."""

    def __init__(
        self,
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float,
        metrics: dict[str, Callable],
        encoder: str = 'google/vit-base-patch16-224-in21k',
        decoder: str = 'google/t5-efficient-base',
        repetition_penalty: float = 1,
        beam_width: int = 1,
    ):
        super().__init__()

        check_encoder_decoder_compatible(encoder, decoder)
        self.encoder = VisualExtractorEncoder(encoder)
        self.decoder = LanguageModelDecoder(decoder)

        self.learning_rate = learning_rate
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.bos_token_id = self.decoder.config.decoder_start_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.repetition_penalty = repetition_penalty
        self.beam_width = beam_width

        self.metrics = metrics
        self.encoder_outputs_to_decoder_tokens = (
            beam_search if self.beam_width > 1 else greedy_search
        )

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
            logits on training or output tokens on validation
        """
        _, imgs, tokens, _ = batch
        encoder_outputs = self.encoder(imgs)

        if self.training:
            return self.decoder(encoder_outputs=encoder_outputs, labels=tokens)
        return self.encoder_outputs_to_decoder_tokens(  # type: ignore[operator]
            self, encoder_outputs
        )

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.FloatTensor:
        """
        Model training step.

        Returns
        -------
        ret
            loss
        """
        output = self(batch)
        loss = output.loss
        self.log('train/loss', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> dict[str, list[str]]:
        """
        Model validation step.

        Returns
        -------
        ret
            report predictions and targets
        """
        output = self(batch)
        _, _, tokens, _ = batch
        return {
            'pred': self.tokenizer.batch_decode(output),
            'target': self.tokenizer.batch_decode(tokens),
        }

    def validation_epoch_end(self, outputs: list[dict[str, list[str]]]):
        """Model validation epoch to log metrics."""
        refs: list[str] = sum([list(x['target']) for x in outputs], [])
        preds: list[str] = sum([list(x['pred']) for x in outputs], [])

        metrics = eval_generated_texts(
            self.metrics, refs, preds, 'val'
        )
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> dict[str, list[str]]:
        """
        Model test step.

        Returns
        -------
        ret
            report predictions and targets
        """
        output = self(batch)
        _, _, tokens, _ = batch
        return {
            'pred': self.tokenizer.batch_decode(output),
            'target': self.tokenizer.batch_decode(tokens),
        }

    def test_epoch_end(self, outputs: list[dict[str, list[str]]]):
        """Model test epoch to log metrics."""
        refs: list[str] = sum([list(x['target']) for x in outputs], [])
        preds: list[str] = sum([list(x['pred']) for x in outputs], [])

        metrics = eval_generated_texts(
            self.metrics, refs, preds, 'test'
        )
        self.log_dict(metrics, on_epoch=True)

    def generate(self, x_ray_images: torch.FloatTensor) -> list[str]:
        """
        Model prediction step.

        Returns
        -------
        ret
            decoded report predictions
        """
        self.eval()
        with torch.no_grad():
            token_ids = self([None, x_ray_images, None, None])

        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def configure_optimizers(self):
        """Configure model optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
