import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from splitter import Splitter


@hydra.main(version_base=None, config_path="../data", config_name="datasets")
def main(cfg: DictConfig) -> None:
    processed_data_dir = cfg.datasets.restaurant_reviews.processed
    reviews_splitter = Splitter(data_dir=processed_data_dir["training"], train_size=0.8)
    X_train, y_train, X_valid, y_valid = reviews_splitter.run()
    logger.info(f"X_train: {X_train.shape}")
    logger.info(f"y_train: {y_train.shape}")
    logger.info(f"X_valid: {X_valid.shape}")
    logger.info(f"y_valid: {y_valid.shape}")

    # Validate stratification
    res = validate_stratif(y_train)
    logger.info(f"y_train value frequency:\n{res}")
    res = validate_stratif(y_valid)
    logger.info(f"y_valid value frequency:\n{res}")


def validate_stratif(y: np.array) -> None:
    unique, counts = np.unique(y[:, 1], return_counts=True)
    return np.asarray((unique, counts)).T


if __name__ == "__main__":
    main()
