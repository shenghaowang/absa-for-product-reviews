import hydra
import torch
from loguru import logger
from model.attention import MultiHeadAttention
from omegaconf import DictConfig, OmegaConf

from train.absa_data import ABSADataModule, ABSAVectorizer
from train.splitter import ABSADataRenderer, ABSADataSplitter


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

    # Prepare datasets
    train_data, valid_data = reviews_splitter.run()
    test_data = reviews_renderer.run()

    # Create data loader
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

    local_batch = next(iter(training_generator))
    local_vectors = local_batch["vectors"].to(device)
    local_labels = local_batch["labels"].to(device)
    logger.info(f"Batch data for training: {local_vectors.size()}")
    logger.info(f"Batch labels for training: {local_labels.size()}")

    # Test the LSTM layer
    hyparams = cfg.model
    seq_len = local_vectors.size()[1]
    packed_input = torch.nn.utils.rnn.pack_padded_sequence(
        input=local_vectors,
        lengths=torch.tensor([seq_len] * hyparams.batch_size),
        batch_first=True,
        enforce_sorted=True,
    )
    logger.info(f"packed_input: {packed_input.data.size()}")

    hyparams = cfg.model
    lstm_layer = torch.nn.LSTM(
        input_size=hyparams.word_vec_dim,
        hidden_size=hyparams.hidden_dim,
        num_layers=hyparams.num_layers,
        batch_first=True,
        bidirectional=True,
    )
    lstm_output, (h, c) = lstm_layer(packed_input)
    logger.info(f"lstm_output: {lstm_output.data.size()}")
    logger.info(f"lstm hidden state: {h.data.size()}")
    logger.info(f"lstm cell state: {c.data.size()}")

    output_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(
        lstm_output, batch_first=True
    )
    logger.info(f"lstm unpacked: {output_unpacked.size()}")

    # Test the attention layer
    multihead_attn = MultiHeadAttention(
        embed_dim=hyparams.hidden_dim * 2, num_heads=2  # bidirectional LSTM
    )
    attn_output = multihead_attn(output_unpacked)
    logger.info(f"multihead_attn: {attn_output.size()}")


if __name__ == "__main__":
    main()
