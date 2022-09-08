import json

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from skmultilearn.model_selection import IterativeStratification


@hydra.main(version_base=None, config_path=".", config_name="datasets")
def main(cfg: DictConfig) -> None:
    processed_data_dir = cfg.datasets.restaurant_reviews.processed
    with open(processed_data_dir["training"], "r") as fin:
        train_data = json.load(fin)

    logger.info(f"Number of reviews loaded: {len(train_data)}")
    df = pd.DataFrame(train_data)
    logger.info(f"\n{df.head(5)}")
    logger.info(f"Valid missing value:\n{df.isna().sum() / df.shape[0]}")

    aspect_cols = ["food", "service", "price", "ambience", "misc"]
    uniq_labels = ["positive", "neutral", "negative", "absent", "conflict"]
    num_uniq_labels = len(uniq_labels)
    df[aspect_cols] = df[aspect_cols].replace(
        to_replace=uniq_labels,
        value=list(range(num_uniq_labels)),
    )
    logger.info(f"\n{df.head(5)}")

    # Split dataset
    X = df["processed_text"].values
    y = df[aspect_cols].values
    logger.info(type(y))
    logger.info(y.shape)
    X_train, X_valid, y_train, y_valid = iterative_train_test_split(
        X, y, train_size=0.8
    )
    logger.info(f"X_train: \n{X_train.shape}")
    logger.info(f"y_train: \n{y_train.shape}")
    logger.info(f"X_valid: \n{X_valid.shape}")
    logger.info(f"y_valid: \n{y_valid.shape}")

    # Validate stratification
    unique, counts = np.unique(y_train[:, 1], return_counts=True)
    logger.info(f"y_train value frequency:\n{np.asarray((unique, counts)).T}")
    unique, counts = np.unique(y_valid[:, 1], return_counts=True)
    logger.info(f"y_valid value frequency:\n{np.asarray((unique, counts)).T}")


def iterative_train_test_split(X, y, train_size):
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'
    """
    stratifier = IterativeStratification(
        n_splits=2,
        order=1,
        sample_distribution_per_fold=[
            1.0 - train_size,
            train_size,
        ],
    )
    train_indices, test_indices = next(stratifier.split(X, y))
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
