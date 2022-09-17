import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from absa_data import ABSADataModule, ABSAVectorizer
from splitter import ABSADataRenderer, ABSADataSplitter

MAX_EPOCHS = 3


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

    # Test data loaders
    device = torch.device("cpu")
    data_module = ABSADataModule(
        vectorizer=ABSAVectorizer(),
        batch_size=cfg.model.batch_size,
        max_seq_len=cfg.features.max_seq_len,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )
    training_generator = data_module.train_dataloader()
    validation_generator = data_module.val_dataloader()
    for epoch in range(MAX_EPOCHS):
        logger.info("\n")
        logger.info(f"==================== Epoch {epoch} ====================")
        # Training
        for local_batch in training_generator:
            local_vectors = local_batch["vectors"].to(device)
            local_labels = local_batch["labels"].to(device)
            logger.info(f"Batch data for training: {local_vectors.size()}")
            logger.info(f"Batch labels for training: {local_labels.size()}")

        # Validation
        with torch.set_grad_enabled(False):
            for local_batch in validation_generator:
                local_vectors = local_batch["vectors"].to(device)
                local_labels = local_batch["labels"].to(device)
                logger.info(f"Batch data for validation: {local_vectors.size()}")
                logger.info(f"Batch labels for validation: {local_labels.size()}")


if __name__ == "__main__":
    main()
