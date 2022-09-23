from typing import List, Tuple

import hydra
import pytorch_lightning as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from absa_data import ABSADataModule, ABSAVectorizer
from model import ABSAClassifier, MultiTaskClassificationModel
from splitter import ABSADataRenderer, ABSADataSplitter

# from torchsummary import summary


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    processed_data_dir = cfg.datasets.restaurant_reviews.processed
    aspects = OmegaConf.to_object(cfg.features.aspects)

    # Initialise datasets
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
    train_data, valid_data = reviews_splitter.run()
    test_data = reviews_renderer.run()

    # Initialise ABSA model
    model = MultiTaskClassificationModel(
        aspects=aspects, num_classes=cfg.features.num_classes, hyparams=cfg.model
    )
    # logger.debug(f"Model architecture:\n{summary(model, (32, 30, 300))}")

    trainer(
        model=model,
        feature_params=cfg.features,
        hyparams=cfg.model,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )


def trainer(
    model: MultiTaskClassificationModel,
    feature_params: DictConfig,
    hyparams: DictConfig,
    train_data: List[Tuple],
    valid_data: List[Tuple],
    test_data: List[Tuple],
) -> None:
    # Create a pytorch trainer
    trainer = pl.Trainer(max_epochs=hyparams.max_epochs, check_val_every_n_epoch=1)

    # Initialize our data loader with the passed vectorizer
    data_module = ABSADataModule(
        vectorizer=ABSAVectorizer(),
        batch_size=hyparams.batch_size,
        max_seq_len=feature_params.max_seq_len,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )

    # Instantiate a new model
    model = ABSAClassifier(
        model,
        aspects=feature_params.aspects,
        label_encoder=feature_params.label_encoder,
        learning_rate=hyparams.learning_rate,
    )

    # Train and validate the model
    trainer.fit(
        model,
        data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    # Test the model
    trainer.test(model, data_module.test_dataloader())

    # Predict on the same test set to show some output
    output = trainer.predict(model, data_module.test_dataloader())

    for i in range(2):
        logger.info("====================")
        logger.info(f"Sentence: {output[1]['sentences'][i]}")
        logger.info(f"Predicted Sentiment: {output[1]['predictions'][i].numpy()}")
        logger.info(f"Actual Label: {output[1]['labels'][i].numpy()}")


if __name__ == "__main__":
    main()
