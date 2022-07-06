"""EncoderDecoder."""

from __future__ import annotations

from typing import Callable

import torch
from pytorch_lightning import LightningModule
from transformers import PreTrainedTokenizer

from vxr.models.decoder import T5Decoder
from vxr.models.encoder import VitEncoder
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
        beam: bool = False,
    ):
        super().__init__()
        self.encoder = VitEncoder(encoder)
        self.decoder = T5Decoder(decoder)

        self.learning_rate = learning_rate
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.bos_token_id = self.decoder.config.decoder_start_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

        self.metrics = metrics
        self.encoder_outputs_to_decoder_tokens = (
            beam_search if beam else greedy_search
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
        return self.encoder_outputs_to_decoder_tokens(self.decoder, encoder_outputs)

    def training_step(self, batch, batch_idx):
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

    def validation_step(self, batch, batch_idx):
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

    def validation_epoch_end(self, outputs):
        """Model validation epoch to log metrics."""
        refs = sum([list(x['target']) for x in outputs], [])
        preds = sum([list(x['pred']) for x in outputs], [])

        metrics = eval_generated_texts(
            self.metrics, refs, preds, 'val'
        )
        self.log_dict(metrics, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
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

    def test_epoch_end(self, outputs):
        """Model test epoch to log metrics."""
        preds = sum([list(x['pred']) for x in outputs], [])
        refs = sum([list(x['target']) for x in outputs], [])

        metrics = eval_generated_texts(
            self.metrics, refs, preds, 'test'
        )
        self.log_dict(metrics, on_epoch=True)

    def generate(self, x_ray_image: torch.FloatTensor) -> list[str]:
        """
        Model prediction step.

        Returns
        -------
        ret
            decoded report predictions
        """
        self.eval()
        with torch.no_grad():
            token_ids = self([None, x_ray_image, None, None])

        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def configure_optimizers(self):
        """Configure model optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
