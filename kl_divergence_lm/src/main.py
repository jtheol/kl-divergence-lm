import nltk

from utils.preprocess import build_dataset
from utils.stats import (
    build_unigram_model,
    build_bigram_model,
    build_trigram_model,
    calc_kl_dataset,
)
from utils.visualize import plot_kl_divergences


def download_nltk_resources():
    """Downloads nltk resources if they are not available."""
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")


if __name__ == "__main__":
    download_nltk_resources()

    wiki_articles, reddit_comments = build_dataset(
        wiki_articles_path="../data/raw/a.parquet",
        reddit_comments_path="../data/raw/reddit_data.csv",
    )

    models = {
        "unigram": build_unigram_model,
        "bigram": build_bigram_model,
        "trigram": build_trigram_model,
    }

    wiki_kl_div = calc_kl_dataset(wiki_articles, models)
    reddit_kl_div = calc_kl_dataset(reddit_comments, models)

    wiki_articles

    plot_kl_divergences(wiki_kl_div, dataset_name="Wikipedia Articles")
    plot_kl_divergences(reddit_kl_div, dataset_name="Reddit Articles")
