from typing import List
from xml.dom import minidom

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path=".", config_name="datasets")
def main(cfg: DictConfig) -> None:
    # logger.info(OmegaConf.to_yaml(cfg))
    logger.debug(OmegaConf.to_container(cfg))
    train_data_dir = cfg.datasets.restaurant_reviews.training
    logger.info(f"Load training data from: {train_data_dir} ...")
    training_reviews = load_reviews(train_data_dir)
    aspect_categories = parse_aspects(training_reviews)
    train_df = parse_reviews(training_reviews, aspect_categories)
    logger.info(f"\n{train_df.head(5)}")


def load_reviews(data_dir: str):
    dom = minidom.parse(data_dir)
    reviews = dom.getElementsByTagName("sentence")
    logger.info(f"There are {len(reviews)} reviews in the training data.")
    return reviews


def parse_aspects(reviews: List[str]):
    uniq_aspect_categories = []
    for review in reviews:
        aspect_container = review.getElementsByTagName("aspectCategories")
        for aspect_item in aspect_container:
            aspect_list = aspect_item.getElementsByTagName("aspectCategory")
            for aspect in aspect_list:
                category = aspect.attributes["category"].value
                if category not in uniq_aspect_categories:
                    uniq_aspect_categories.append(category)
    logger.info(f"Number of aspects found: {uniq_aspect_categories}")
    logger.info(uniq_aspect_categories)
    return uniq_aspect_categories


def parse_reviews(reviews: List[str], uniq_aspect_categories: List[str]):
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


if __name__ == "__main__":
    main()
