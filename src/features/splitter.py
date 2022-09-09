from typing import List, Tuple

import numpy as np
import pandas as pd
from feature_cfg import ASPECT_COLS, LABEL_ENCODER, REVIEW_COL
from loguru import logger
from skmultilearn.model_selection import IterativeStratification


class ABSADataRenderer:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.reviews_df = None

    def load_data(self) -> None:
        self.reviews_df = pd.read_parquet(self.data_dir, engine="pyarrow")
        logger.info(f"Number of reviews loaded: {len(self.reviews_df)}")
        logger.info(f"Preview of dataset:\n{self.reviews_df.head(5)}")

    def check_num_words(self) -> None:
        reviews_df = self.reviews_df.copy(deep=True)
        reviews_df["num_words"] = reviews_df["text"].apply(
            lambda text: len(text.split(" "))
        )
        logger.info(f"Length of reviews:\n{reviews_df['num_words'].describe()}")

    def encode_labels(self) -> None:
        self.reviews_df[ASPECT_COLS] = self.reviews_df[ASPECT_COLS].replace(
            LABEL_ENCODER
        )
        logger.info(f"\n{self.reviews_df.head(5)}")

    @staticmethod
    def render_data(X, y) -> List[Tuple]:
        data = []
        for idx, review in enumerate(X):
            data.append((y[idx], review))

        return data

    def run(self) -> List[Tuple]:
        self.load_data()
        self.encode_labels()
        X = self.reviews_df[REVIEW_COL].values
        y = self.reviews_df[ASPECT_COLS].values

        return self.render_data(X, y)


class ABSADataSplitter(ABSADataRenderer):
    def __init__(self, data_dir: str, train_size: float) -> None:
        super().__init__(data_dir)
        self.train_size = train_size

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

    def run(self) -> Tuple[List, List]:
        self.load_data()
        self.check_num_words()
        self.encode_labels()
        X = self.reviews_df[REVIEW_COL].values
        y = self.reviews_df[ASPECT_COLS].values
        train_indices, valid_indices = self.iterative_train_test_split(X, y)
        train_data = self.render_data(X[train_indices], y[train_indices])
        valid_data = self.render_data(X[valid_indices], y[valid_indices])

        return train_data, valid_data
