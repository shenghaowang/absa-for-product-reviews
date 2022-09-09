import hydra
from loguru import logger
from omegaconf import DictConfig
from splitter import ABSADataRenderer, ABSADataSplitter


@hydra.main(version_base=None, config_path="../data", config_name="datasets")
def main(cfg: DictConfig) -> None:
    processed_data_dir = cfg.datasets.restaurant_reviews.processed
    reviews_splitter = ABSADataSplitter(
        data_dir=processed_data_dir["training"], train_size=0.8
    )
    reviews_renderer = ABSADataRenderer(data_dir=processed_data_dir["test"])

    # Prepare training and validation data
    train_data, valid_data = reviews_splitter.run()
    logger.info(f"Volume of training data: {len(train_data)}")
    logger.info(f"Preview of training data:\n{train_data[:3]}")
    logger.info(f"Volume of validation data: {len(valid_data)}")
    logger.info(f"Preview of validation data:\n{valid_data[:3]}")

    # Prepare test data
    test_data = reviews_renderer.run()
    logger.info(f"Volume of test data: {len(test_data)}")
    logger.info(f"Preview of test data:\n{test_data[:3]}")


if __name__ == "__main__":
    main()
