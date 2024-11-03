from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from skmultilearn.model_selection import IterativeStratification


class ABSADataRenderer:
    def __init__(
        self,
        data_dir: str,
        aspect_cols: List[str],
        review_col: str,
        label_encoder: Dict[str, int],
    ) -> None:
        """Split test data into review and aspect labels

        Parameters
        ----------
        data_dir : str
            directory of the processed test data
        aspect_cols : List[str]
            data columns which provide the aspect labels
        review_col : str
            data column which provides the processed review
        label_encoder : Dict[str, int]
            mapping between polarity label and integer labels
        """
        self.data_dir = data_dir
        self.aspect_cols = aspect_cols
        self.review_col = review_col
        self.label_encoder = label_encoder
        self.reviews_df = None

    def load_data(self) -> None:
        """Load processed review data in a dataframe"""
        self.reviews_df = pd.read_parquet(self.data_dir, engine="pyarrow")
        logger.info(f"Number of reviews loaded: {len(self.reviews_df)}")
        logger.info(f"Preview of dataset:\n{self.reviews_df.head(5)}")

    def check_num_words(self) -> None:
        """Generate descriptive statistics for the length of reviews"""
        reviews_df = self.reviews_df.copy(deep=True)
        reviews_df["num_words"] = reviews_df["text"].apply(
            lambda text: len(text.split(" "))
        )
        logger.info(f"Length of reviews:\n{reviews_df['num_words'].describe()}")

    def encode_labels(self) -> None:
        """Encode the polarity labels for different aspects using integers"""
        self.reviews_df[self.aspect_cols] = self.reviews_df[self.aspect_cols].replace(
            self.label_encoder
        )

        logger.info(f"\n{self.reviews_df.head(5)}")

    @staticmethod
    def render_data(X: np.array, y: np.array) -> List[Tuple]:
        """Generate pairs of aspect labels & reviews

        Parameters
        ----------
        X : np.array
            processed reviews
        y : np.array
            polarity labels of different aspects

        Returns
        -------
        List[Tuple]
            Paired labels and reviews
        """
        data = []
        for idx, review in enumerate(X):
            data.append((y[idx], review))

        return data

    def run(self) -> List[Tuple]:
        self.load_data()
        self.encode_labels()
        X = self.reviews_df[self.review_col].to_numpy()
        y = self.reviews_df[self.aspect_cols].to_numpy()

        return self.render_data(X, y)


class ABSADataSplitter(ABSADataRenderer):
    def __init__(
        self,
        data_dir: str,
        aspect_cols: List[str],
        review_col: str,
        label_encoder: Dict[str, int],
        train_size: float,
    ) -> None:
        """Split data into review and aspect labels
        for training and validation

        Parameters
        ----------
        data_dir : str
            directory of the processed data
        aspect_cols : List[str]
            data columns which provide the aspect labels
        review_col : str
            data column which provides the processed review
        label_encoder : Dict[str, int]
            mapping between polarity label and integer labels
        train_size : float
            proportion of the training data
        """
        super().__init__(data_dir, aspect_cols, review_col, label_encoder)
        self.train_size = train_size

    def iterative_train_test_split(
        self, X: np.array, y: np.array
    ) -> Tuple[np.array, np.array]:
        """Custom iterative train test split which maintains the
        distribution of labels in the training and test sets

        Parameters
        ----------
        X : np.array
            features array
        y : np.array
            labels array

        Returns
        -------
        Tuple[np.array, np.array]
            indices of training and test examples
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
        X = self.reviews_df[self.review_col].values
        y = self.reviews_df[self.aspect_cols].values
        train_indices, valid_indices = self.iterative_train_test_split(X, y)
        train_data = self.render_data(X[train_indices], y[train_indices])
        valid_data = self.render_data(X[valid_indices], y[valid_indices])

        return train_data, valid_data
