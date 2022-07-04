"""EncoderDecoder."""

from __future__ import annotations

import evaluate
import torch
from pytorch_lightning import LightningModule
from transformers import PreTrainedTokenizer

from vxr.models.decoder import T5Decoder
from vxr.models.encoder import VitEncoder


class XrayReportGeneration(LightningModule):
    """Model for generating x-ray reports from images."""

    def __init__(
        self,
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float
    ):
        super().__init__()
        self.encoder = VitEncoder()
        self.decoder = T5Decoder()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.learning_rate = learning_rate

        self.metrics = {
            'BLEU-1': evaluate.load('bleu'),
            'BLEU-2': evaluate.load('bleu'),
            'BLEU-3': evaluate.load('bleu'),
            'BLEU-4': evaluate.load('bleu'),
            'METEOR': evaluate.load('meteor'),
        }

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
            return self.decoder(
                encoder_outputs=encoder_outputs,
                labels=tokens
            )
        else:
            return self._encoder_outputs_to_decoder_tokens(encoder_outputs)

    def _encoder_outputs_to_decoder_tokens(
        self, encoder_outputs: torch.FloatTensor
    ) -> torch.LongTensor:
        """
        Generate report tokens.

        Parameters
        ----------
        encoder_outputs
            features extracted by the encoder

        Returns
        -------
        ret
            generated report token ids
        """
        decoder_tokens = torch.full(
            (len(encoder_outputs.last_hidden_state), 1),
            self.decoder.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device
        )

        for _ in range(self.max_length):
            outputs = self.decoder(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_tokens
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
            if torch.eq(next_token_id[:, -1], self.eos_token_id).all():
                break
            decoder_tokens = torch.cat([decoder_tokens, next_token_id], dim=-1)

        return decoder_tokens

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
            'target': self.tokenizer.batch_decode(tokens)
        }

    def validation_epoch_end(self, outputs):
        """Model validation epoch to log metrics."""
        preds = sum([list(x['pred']) for x in outputs], [])
        refs = sum([list(x['target']) for x in outputs], [])

        metrics = {
            'val/BLEU-1': self.metrics['BLEU-1'].compute(
                references=preds, predictions=refs, max_order=1
            )['bleu'],
            'val/BLEU-2': self.metrics['BLEU-2'].compute(
                references=preds, predictions=refs, max_order=2
            )['bleu'],
            'val/BLEU-3': self.metrics['BLEU-3'].compute(
                references=preds, predictions=refs, max_order=3
            )['bleu'],
            'val/BLEU-4': self.metrics['BLEU-4'].compute(
                references=preds, predictions=refs, max_order=4
            )['bleu'],
            'val/METEOR': self.metrics['METEOR'].compute(
                references=preds, predictions=refs
            )['meteor'],
        }
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
            'target': self.tokenizer.batch_decode(tokens)
        }

    def test_epoch_end(self, outputs):
        """Model test epoch to log metrics."""
        preds = sum([list(x['pred']) for x in outputs], [])
        refs = sum([list(x['target']) for x in outputs], [])

        metrics = {
            'test/BLEU-1': self.metrics['BLEU-1'].compute(
                references=preds, predictions=refs, max_order=1
            )['bleu'],
            'test/BLEU-2': self.metrics['BLEU-2'].compute(
                references=preds, predictions=refs, max_order=2
            )['bleu'],
            'test/BLEU-3': self.metrics['BLEU-3'].compute(
                references=preds, predictions=refs, max_order=3
            )['bleu'],
            'test/BLEU-4': self.metrics['BLEU-4'].compute(
                references=preds, predictions=refs, max_order=4
            )['bleu'],
            'test/METEOR': self.metrics['METEOR'].compute(
                references=preds, predictions=refs
            )['meteor'],
        }
        self.log_dict(metrics, on_epoch=True)

    def generate_report(self, x_ray_image: torch.FloatTensor) -> list[str]:
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

        return self.tokenizer.batch_decode(token_ids)

    def configure_optimizers(self):
        """Configure model optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
