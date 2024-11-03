from typing import List

import numpy as np
from loguru import logger


class Predictor:
    """
    Class to create baseline predictions and compute the accuracy
    of sentiment classification
    """

    @staticmethod
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

    @staticmethod
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
