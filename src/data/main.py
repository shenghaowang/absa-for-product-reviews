from pathlib import Path

import cleantext
import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.review_preprocessor import ReviewPreprocessor


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.debug(OmegaConf.to_container(cfg))
    raw_data_dir = cfg.datasets.restaurant_reviews.raw
    processed_data_dir = cfg.datasets.restaurant_reviews.processed

    review_preprocessor = ReviewPreprocessor()

    for ds_type in ["training", "test"]:
        logger.info(f"Load {ds_type} data from: {raw_data_dir[ds_type]} ...")
        raw_reviews = review_preprocessor.load_reviews(raw_data_dir[ds_type])
        aspect_categories = review_preprocessor.parse_aspects(raw_reviews)
        reviews_df = review_preprocessor.parse_reviews(raw_reviews, aspect_categories)

        # Clean up reviews if requested
        if cfg.datasets.clean:
            reviews_df["text"] = reviews_df["text"].apply(
                cleantext.clean,
                extra_spaces=True,
                lowercase=True,
                stopwords=True,
                numbers=True,
                punct=True,
            )

            # Remove which are blank after cleaning
            reviews_df = reviews_df[
                reviews_df["text"].apply(lambda x: len(x.strip()) > 0)
            ]
            logger.info(f"{len(reviews_df)} reviews remain after text cleaning.")

        # Check distribution of labels by different aspects
        labels = review_preprocessor.count_labels(reviews_df, aspect_categories)
        logger.info(f"Distribution of labels by aspects: \n{labels}")

        # Export processed reviews
        reviews_df.rename(columns={"anecdotes/miscellaneous": "misc"}, inplace=True)
        logger.info(f"\n{reviews_df.head(5)}")
        table = pa.Table.from_pandas(reviews_df, preserve_index=True)

        output_dir = Path(processed_data_dir[ds_type]).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, processed_data_dir[ds_type])

        logger.info(f"Processed reviews written to {processed_data_dir[ds_type]}")


if __name__ == "__main__":
    main()
