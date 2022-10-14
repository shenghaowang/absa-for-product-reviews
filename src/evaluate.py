from typing import List

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from absa_data import ABSADataModule, ABSAVectorizer
from model import ABSAClassifier, MultiTaskClassificationModel
from splitter import ABSADataRenderer, ABSADataSplitter


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    processed_data_dir = cfg.datasets.restaurant_reviews.processed
    aspects = OmegaConf.to_object(cfg.features.aspects)

    reviews_splitter = ABSADataSplitter(
        data_dir=processed_data_dir["training"],
        train_size=0.8,
        review_col=cfg.features.review_col,
        aspect_cols=aspects,
        label_encoder=cfg.features.label_encoder,
    )
    reviews_renderer = ABSADataRenderer(
        data_dir=processed_data_dir["test"],
        review_col=cfg.features.review_col,
        aspect_cols=aspects,
        label_encoder=cfg.features.label_encoder,
    )

    # Load datasets
    train_data, valid_data = reviews_splitter.run()
    test_data = reviews_renderer.run()
    logger.info(f"Volume of test data: {len(test_data)}")
    logger.info(f"Preview of test data:\n{test_data[:3]}")

    y_true = np.array([labels for labels, review in test_data])
    logger.debug(f"Groundtruth labels: {y_true.shape}")

    # Compute accuracy for majority prediction
    y_majority_pred = get_majority_prediction(y_true, cfg.features.num_aspects)
    logger.info("Compute accuracy for the majority predictions:")
    compute_accuracy(
        y_true=y_true,
        y_pred=y_majority_pred,
        aspects=aspects,
        absent=cfg.features.label_encoder.absent,
    )

    # Load fitted ABSA model
    model = ABSAClassifier(
        model=MultiTaskClassificationModel(
            aspects=aspects, num_classes=cfg.features.num_classes, hyparams=cfg.model
        ),
        aspects=aspects,
        label_encoder=cfg.features.label_encoder,
        learning_rate=cfg.model.learning_rate,
        class_weights=cfg.model.class_weights,
    )
    model.load_state_dict(torch.load(cfg.model_file))
    model.eval()

    # Make test predictions
    data_module = ABSADataModule(
        vectorizer=ABSAVectorizer(),
        batch_size=cfg.model.batch_size,
        max_seq_len=cfg.features.max_seq_len,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )
    trainer = pl.Trainer(max_epochs=cfg.model.max_epochs, check_val_every_n_epoch=1)
    output = trainer.predict(model, data_module.test_dataloader())

    logger.debug("Compute accuracy for the ABSA model:")
    test_labels = torch.cat([pred["labels"] for pred in output])
    test_preds = torch.cat([pred["predictions"] for pred in output])
    compute_accuracy(
        y_true=test_labels.numpy(),
        y_pred=test_preds.numpy(),
        aspects=aspects,
        absent=cfg.features.label_encoder.absent,
    )


def get_majority_prediction(
    y_true: np.array, num_aspects: int, absent: int = 0
) -> np.array:
    """Create baseline predictions based on majority voting

    Parameters
    ----------
    y_true : np.array
        groudtruth label across aspects
    num_aspects : int
        number of aspects
    absent : int, optional
        encoded label to be exempted from the accuracy calculation,
        by default 0

    Returns
    -------
    np.array
        predicted labels based on majority voting
    """
    y_majority_pred = []
    for idx in range(num_aspects):
        label_counts = np.bincount(y_true[:, idx])
        labels_sorted_by_freq = np.argsort(label_counts)[::-1]
        if labels_sorted_by_freq[0] != absent:
            majority_label = labels_sorted_by_freq[0]
        else:
            majority_label = labels_sorted_by_freq[1]

        aspect_majority_pred = np.full(y_true[:, idx].shape, majority_label)
        y_majority_pred.append(aspect_majority_pred)

    return np.vstack(y_majority_pred).T


def compute_accuracy(
    y_true: np.array, y_pred: np.array, aspects: List[str], absent: int = 0
) -> None:
    """Compute per aspect and overall accuracy

    Parameters
    ----------
    y_true : np.array
        groundtruth labels across aspects
    y_pred : np.array
        predicted labels across aspects
    aspects : List[str]
        names of the aspects
    absent : int, optional
        encoded label to be exempted from the accuracy calculation,
        by default 0
    """

    assert (
        y_true.shape == y_pred.shape
    ), "Prediction and groundtruth labels do not do not share the same shape."

    agg_num_pred = 0
    agg_num_correct_pred = 0
    for idx, aspect in enumerate(aspects):
        y_true_aspect = y_true[:, idx]
        y_pred_aspect = y_pred[:, idx]

        num_pred = np.clip(y_true_aspect, 0, 1).sum()
        num_correct_pred = (
            ((y_true_aspect == y_pred_aspect) & (y_true_aspect != absent))
            .astype(int)
            .sum()
        )
        logger.info(
            f"Accuracy for {aspect}: {num_correct_pred} / {num_pred}"
            + f" = {round(num_correct_pred / num_pred, 4)}"
        )

        agg_num_pred += num_pred
        agg_num_correct_pred += num_correct_pred

    logger.info(
        f"Overall accuracy: {agg_num_correct_pred} / {agg_num_pred}"
        + f" = {round(agg_num_correct_pred / agg_num_pred, 4)}"
    )


if __name__ == "__main__":
    main()
