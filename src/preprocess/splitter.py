import json
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from skmultilearn.model_selection import IterativeStratification

REVIEW_COL = "processed_text"
ASPECT_COLS = ["food", "service", "price", "ambience", "misc"]
UNIQ_LABELS = ["positive", "neutral", "negative", "absent", "conflict"]


class Splitter:
    def __init__(self, data_dir: str, train_size: float) -> None:
        self.data_dir = data_dir
        self.train_size = train_size
        self.reviews_df = None

    def load_data(self) -> None:
        with open(self.data_dir, "r") as fin:
            data = json.load(fin)

        logger.info(f"Number of reviews loaded: {len(data)}")
        self.reviews_df = pd.DataFrame(data)

    def encode_labels(self) -> None:
        num_uniq_labels = len(UNIQ_LABELS)
        self.reviews_df[ASPECT_COLS] = self.reviews_df[ASPECT_COLS].replace(
            to_replace=UNIQ_LABELS,
            value=list(range(num_uniq_labels)),
        )
        logger.info(f"\n{self.reviews_df.head(5)}")

    def iterative_train_test_split(
        self, X: np.array, y: np.array
    ) -> Tuple[np.array, np.array]:
        """Custom iterative train test split which
        'maintains balanced representation with respect
        to order-th label combinations.'
        """
        stratifier = IterativeStratification(
            n_splits=2,
            order=1,
            sample_distribution_per_fold=[
                1.0 - self.train_size,
                self.train_size,
            ],
        )
        train_indices, test_indices = next(stratifier.split(X, y))
        logger.info(f"Training indices: {train_indices[:10]}")
        logger.info(f"Test indices: {test_indices[:10]}")

        return train_indices, test_indices

    def run(self):
        self.load_data()
        self.encode_labels()
        X = self.reviews_df[REVIEW_COL].values
        y = self.reviews_df[ASPECT_COLS].values
        train_indices, valid_indices = self.iterative_train_test_split(X, y)

        return (X[train_indices], y[train_indices], X[valid_indices], y[valid_indices])
