from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics


class ABSAClassifier(pl.LightningModule):
    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.params = params
        self.f1 = torchmetrics.F1Score()

    def forward(self, x, x_len):
        return self.model(x, x_len)

    def training_step(self, batch, batch_idx):
        x = batch["vectors"]
        x_len = batch["vectors_length"]
        y = batch["labels"]
        y_hat = self(x, x_len)
        loss = F.cross_entropy(y_hat, y, reduction="mean")
        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x = batch["vectors"]
        x_len = batch["vectors_length"]
        y = batch["labels"]
        y_hat = self(x, x_len)
        loss = F.cross_entropy(y_hat, y, reduction="mean")
        predictions = torch.argmax(y_hat, dim=1)
        self.log_dict(
            {
                "val_loss": loss,
                "val_f1": self.f1(predictions, y),
            },
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_nb):
        x = batch["vectors"]
        x_len = batch["vectors_length"]
        y = batch["labels"]
        y_hat = self(x, x_len)
        loss = F.cross_entropy(y_hat, y, reduction="mean")
        predictions = torch.argmax(y_hat, dim=1)
        self.log_dict(
            {
                "test_loss": loss,
                "test_f1": self.f1(predictions, y),
            },
            prog_bar=True,
        )
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
        return optimizer


class MultiTaskClassificationModel(torch.nn.Module):
    def __init__(
        self,
        aspects: List[str],
        word_vec_dimension: int,
        # num_aspects: int,
        num_classes: int,
        params,
    ):
        super().__init__()
        self.backbone = torch.nn.LSTM(
            input_size=word_vec_dimension,
            hidden_size=params.hidden_dim,
            num_layers=params.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Define task specific layers
        self.heads = torch.nn.ModuleList([])
        for aspect in aspects:
            module_name = f"h_{aspect}"
            module = torch.nn.Sequential(
                torch.nn.Linear(params.hidden_dim, params.hidden_dim / 2),
                torch.nn.Dropout(params.dropout),
                torch.nn.Linear(params.hidden_dim / 2, num_classes),
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
