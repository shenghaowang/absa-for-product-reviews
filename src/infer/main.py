import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from absa.model import ABSAClassifier, MultiTaskClassificationModel
from infer.predictor import Predictor
from train.absa_data import ABSADataModule, ABSAVectorizer
from train.splitter import ABSADataRenderer, ABSADataSplitter


@hydra.main(version_base=None, config_path="../config", config_name="config")
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

    # Load datasets
    train_data, valid_data = reviews_splitter.run()
    test_data = reviews_renderer.run()
    logger.info(f"Volume of test data: {len(test_data)}")
    logger.info(f"Preview of test data:\n{test_data[:3]}")

    y_true = np.array([labels for labels, review in test_data])
    logger.debug(f"Groundtruth labels: {y_true.shape}")

    # Compute accuracy for majority prediction
    predictor = Predictor()

    y_majority_pred = predictor.get_majority_prediction(
        y_true, cfg.features.num_aspects
    )
    logger.info("Compute accuracy for the majority predictions:")
    predictor.compute_accuracy(
        y_true=y_true,
        y_pred=y_majority_pred,
        aspects=aspects,
        absent=cfg.features.label_encoder.absent,
    )

    # Load fitted ABSA model
    model = ABSAClassifier(
        model=MultiTaskClassificationModel(
            aspects=aspects, num_classes=cfg.features.num_classes, hyparams=cfg.model
        ),
        aspects=aspects,
        label_encoder=cfg.features.label_encoder,
        learning_rate=cfg.model.learning_rate,
        class_weights=cfg.model.class_weights,
    )
    model.load_state_dict(torch.load(cfg.model_file))
    model.eval()

    # Make test predictions
    data_module = ABSADataModule(
        vectorizer=ABSAVectorizer(),
        batch_size=cfg.model.batch_size,
        max_seq_len=cfg.features.max_seq_len,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )
    trainer = pl.Trainer(max_epochs=cfg.model.max_epochs, check_val_every_n_epoch=1)
    output = trainer.predict(model, data_module.test_dataloader())

    logger.debug("Compute accuracy for the ABSA model:")
    test_labels = torch.cat([pred["labels"] for pred in output])
    test_preds = torch.cat([pred["predictions"] for pred in output])
    predictor.compute_accuracy(
        y_true=test_labels.numpy(),
        y_pred=test_preds.numpy(),
        aspects=aspects,
        absent=cfg.features.label_encoder.absent,
    )


if __name__ == "__main__":
    main()
