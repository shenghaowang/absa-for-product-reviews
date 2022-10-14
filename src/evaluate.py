from typing import List

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from splitter import ABSADataRenderer


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    processed_data_dir = cfg.datasets.restaurant_reviews.processed
    aspects = OmegaConf.to_object(cfg.features.aspects)
    reviews_renderer = ABSADataRenderer(
        data_dir=processed_data_dir["test"],
        review_col=cfg.features.review_col,
        aspect_cols=aspects,
        label_encoder=cfg.features.label_encoder,
    )

    # Load test data
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


def get_majority_prediction(
    y_true: np.array, num_aspects: int, absent: int = 0
) -> np.array:
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
        logger.info(f"Accuracy for {aspect}: {round(num_correct_pred / num_pred, 4)}")

        agg_num_pred += num_pred
        agg_num_correct_pred += num_correct_pred

    logger.info(f"Overall accuracy: {round(agg_num_correct_pred / agg_num_pred, 4)}")


if __name__ == "__main__":
    main()
