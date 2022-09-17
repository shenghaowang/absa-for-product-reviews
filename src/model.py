from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from loguru import logger
from omegaconf import DictConfig


class ABSAClassifier(pl.LightningModule):
    def __init__(
        self,
        model,
        aspects: List[str],
        label_encoder: Dict[str, int],
        learning_rate: float,
    ):
        super().__init__()
        self.model = model
        self.aspects = aspects
        self.label_encoder = label_encoder
        self.num_aspects = len(self.aspects)
        self.learning_rate = learning_rate
        self.acc = torchmetrics.Accuracy()

    def forward(self, x, x_len):
        return self.model(x, x_len)

    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform labels to one-hot vectors
        x = batch["vectors"]
        x_len = batch["vectors_length"]
        y = batch["labels"]  # One-hot encoding is not required

        # Perform prediction and calculate loss and F1 score
        y_hat = self(x, x_len)
        logger.debug(f"y: {y.size()}")
        logger.debug(f"y_hat: {y_hat.size()}")
        agg_loss = 0
        total_examples = 0
        correct_examples = 0
        for idx, _ in enumerate(self.aspects):
            prob_start_idx = self.num_aspects * idx
            prob_end_idx = self.num_aspects * (idx + 1)

            # Skip data points with the "absent" label
            valid_label_ids = np.where(y[:, idx] != self.label_encoder["absent"])
            loss = F.cross_entropy(
                y_hat[valid_label_ids, prob_start_idx:prob_end_idx],
                y[valid_label_ids, idx],
                reduction="mean",
            )
            agg_loss += loss

            predictions = torch.argmax(
                y_hat[valid_label_ids, prob_start_idx:prob_end_idx], dim=1
            )
            correct_examples += torch.sum(predictions == y[valid_label_ids, idx])
            total_examples += len(valid_label_ids)
            acc = correct_examples / total_examples

        # Logging
        self.log_dict(
            {
                f"{mode}_loss": agg_loss,
                f"{mode}_acc": acc,
            },
            prog_bar=True,
        )
        return agg_loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, "train")
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self._calculate_loss(batch, "val")
        return loss

    def test_step(self, batch, batch_nb):
        loss = self._calculate_loss(batch, "test")
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch["vectors"]
        x_len = batch["vectors_length"]
        y_hat = self.model(x, x_len)
        predictions = torch.argmax(y_hat, dim=1)
        return {
            "logits": y_hat,
            "predictions": predictions,
            "labels": batch["labels"],
            "sentences": batch["sentences"],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class MultiTaskClassificationModel(torch.nn.Module):
    def __init__(
        self,
        aspects: List[str],
        num_classes: int,
        hyparams: DictConfig,
    ):
        super().__init__()
        self.aspects = aspects
        self.backbone = torch.nn.LSTM(
            input_size=hyparams.word_vec_dim,
            hidden_size=hyparams.hidden_dim,
            num_layers=hyparams.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Define task specific layers
        self.heads = torch.nn.ModuleList([])
        for aspect in aspects:
            module_name = f"h_{aspect}"
            module = torch.nn.Sequential(
                torch.nn.Linear(hyparams.hidden_dim, int(hyparams.hidden_dim / 2)),
                torch.nn.Dropout(hyparams.dropout),
                torch.nn.Linear(int(hyparams.hidden_dim / 2), num_classes),
            )
            setattr(self, module_name, module)

    def forward(self, batch, batch_len):
        """Projection from word_vec_dim to n_classes
        Batch in is shape (batch_size, max_seq_len, word_vector_dim)

        Batch out is shape (batch, num_classes)
        """

        # This deals with variable length sentences. Sorted works faster.
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            batch, batch_len, batch_first=True, enforce_sorted=True
        )
        lstm_output, (h, c) = self.backbone(packed_input)
        hs = [getattr(self, f"h_{aspect}")(h) for aspect in self.aspects]
        return torch.cat(hs, axis=1)
