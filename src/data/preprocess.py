import json
import re
from typing import List
from xml.dom import minidom

import hydra
import nltk
import pandas as pd
import spacy
from loguru import logger
from nltk.corpus import stopwords
from omegaconf import DictConfig, OmegaConf

nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


@hydra.main(version_base=None, config_path=".", config_name="datasets")
def main(cfg: DictConfig) -> None:
    # logger.info(OmegaConf.to_yaml(cfg))
    logger.debug(OmegaConf.to_container(cfg))
    raw_data_dir = cfg.datasets.restaurant_reviews.raw
    processed_data_dir = cfg.datasets.restaurant_reviews.processed

    processed_data_cols = [
        "processed_text",
        "food",
        "service",
        "price",
        "ambience",
        "misc",
    ]
    for ds_type in ["training", "test"]:
        logger.info(f"Load {ds_type} data from: {raw_data_dir[ds_type]} ...")
        raw_reviews = load_reviews(raw_data_dir[ds_type])
        aspect_categories = parse_aspects(raw_reviews)
        reviews_df = parse_reviews(raw_reviews, aspect_categories)

        # Check distribution of labels by different aspects
        labels = count_labels(reviews_df, aspect_categories)
        logger.info(f"Distribution of labels by aspects: \n{labels}")

        # Remove stopwords, punctuations and normalise the text
        reviews_df.rename(columns={"anecdotes/miscellaneous": "misc"}, inplace=True)
        logger.info("Cleaning reviews ...")
        reviews_df.loc[:, "processed_text"] = reviews_df["text"].apply(clean_review)
        reviews_df = reviews_df[processed_data_cols]
        logger.info(f"\n{reviews_df.head(5)}")

        # Export processed reviews
        review_objs = reviews_df.to_dict("records")
        with open(processed_data_dir[ds_type], "w") as fout:
            json.dump(review_objs, fout, indent=4)

        logger.info(f"Processed reviews written to {processed_data_dir[ds_type]}")


def load_reviews(data_dir: str):
    """Read restaurant reviews from .xml file"""
    dom = minidom.parse(data_dir)
    reviews = dom.getElementsByTagName("sentence")
    logger.info(f"There are {len(reviews)} reviews in the training data.")
    return reviews


def parse_aspects(reviews: List[str]):
    """Parse aspects from all the reviews"""
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


def parse_reviews(reviews: List[str], uniq_aspect_categories: List[str]):
    """"""
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


def count_labels(data: pd.DataFrame, aspect_categories: List[str]):
    """
    Check the distribution of labels under different aspects

    """

    label_counts = pd.DataFrame()
    for aspect in aspect_categories:
        label_counts[aspect] = data[aspect].value_counts()

    return label_counts


def clean_review(text: str):
    """Clean text"""

    def rm_punctuation(text):
        return re.sub(r"[^\w\s]", "", text)

    def rm_stopwords(text):
        stop = stopwords.words("english")
        return " ".join(word for word in text.split() if word not in stop)

    def lemmatize(text):
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])

    text = text.lower()
    text = rm_punctuation(text)
    text = rm_stopwords(text)
    text = lemmatize(text)

    return text


if __name__ == "__main__":
    main()
