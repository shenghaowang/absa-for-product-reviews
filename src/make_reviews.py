from typing import List
from xml.dom import minidom

import cleantext
import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.debug(OmegaConf.to_container(cfg))
    raw_data_dir = cfg.datasets.restaurant_reviews.raw
    processed_data_dir = cfg.datasets.restaurant_reviews.processed

    for ds_type in ["training", "test"]:
        logger.info(f"Load {ds_type} data from: {raw_data_dir[ds_type]} ...")
        raw_reviews = load_reviews(raw_data_dir[ds_type])
        aspect_categories = parse_aspects(raw_reviews)
        reviews_df = parse_reviews(raw_reviews, aspect_categories)

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
        labels = count_labels(reviews_df, aspect_categories)
        logger.info(f"Distribution of labels by aspects: \n{labels}")

        # Export processed reviews
        reviews_df.rename(columns={"anecdotes/miscellaneous": "misc"}, inplace=True)
        logger.info(f"\n{reviews_df.head(5)}")
        table = pa.Table.from_pandas(reviews_df, preserve_index=True)
        pq.write_table(table, processed_data_dir[ds_type])

        logger.info(f"Processed reviews written to {processed_data_dir[ds_type]}")


def load_reviews(data_dir: str) -> List[minidom.Element]:
    """Read restaurant reviews from .xml file

    Parameters
    ----------
    data_dir : str
        directory of the raw data file.

    Returns
    -------
    reviews : List[minidom.Element]
        a list of review entities
    """
    dom = minidom.parse(data_dir)
    reviews = dom.getElementsByTagName("sentence")
    logger.info(f"There are {len(reviews)} reviews in the training data.")
    return reviews


def parse_aspects(reviews: List[minidom.Element]) -> List[str]:
    """Extract unique aspects from the review entities

    Parameters
    ----------
    reviews : List[minidom.Element]
        list of review objects

    Returns
    -------
    uniq_aspect_categories : List[str]
        list of unique aspect names
    """
    uniq_aspect_categories = []
    for review in reviews:
        aspect_container = review.getElementsByTagName("aspectCategories")
        for aspect_item in aspect_container:
            aspect_list = aspect_item.getElementsByTagName("aspectCategory")
            for aspect in aspect_list:
                category = aspect.attributes["category"].value
                if category not in uniq_aspect_categories:
                    uniq_aspect_categories.append(category)

    logger.info(f"Number of aspects found: {len(uniq_aspect_categories)}")
    logger.info(uniq_aspect_categories)

    return uniq_aspect_categories


def parse_reviews(
    reviews: List[minidom.Element], uniq_aspect_categories: List[str]
) -> pd.DataFrame:
    """Collect the aspect labels of restaurant reviews
    in a dataframe

    Parameters
    ----------
    reviews : List[minidom.Element]
        list of review objects
    uniq_aspect_categories : List[str]
        list of aspect names

    Returns
    -------
    pd.DataFrame
        a dataframe which contains the review and aspect polarity
    """
    data = []
    for review in reviews:
        obj = {}
        text_list = review.getElementsByTagName("text")
        for text_item in text_list:
            obj["text"] = text_item.firstChild.data

        aspect_container = review.getElementsByTagName("aspectCategories")
        for category in uniq_aspect_categories:
            obj[category] = "absent"
        for aspect_item in aspect_container:
            aspect_list = aspect_item.getElementsByTagName("aspectCategory")
            for aspect in aspect_list:
                category = aspect.attributes["category"].value
                polarity = aspect.attributes["polarity"].value
                obj[category] = polarity

        data.append(obj)

    return pd.DataFrame(data)


def count_labels(data: pd.DataFrame, aspect_categories: List[str]) -> pd.DataFrame:
    """Check the distribution of polarity labels
    under different aspects

    Parameters
    ----------
    data : pd.DataFrame

    aspect_categories : List[str]
        list of unique aspect names

    Returns
    -------
    label_counts : pd.DataFrame
        count of reviews wrt. aspect x polarity label
    """

    label_counts = pd.DataFrame()
    for aspect in aspect_categories:
        label_counts[aspect] = data[aspect].value_counts()

    return label_counts


if __name__ == "__main__":
    main()
