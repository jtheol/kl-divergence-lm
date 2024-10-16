import re
from typing import Tuple
import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocess_articles(text):
    """Preprocess Wikipedia article text.

    Args:
        text (str): The Wikipedia article text to preprocess.

    Returns:
        list: A list of preprocessed tokens
    """

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


def build_dataset(
    wiki_articles_path: str,
    reddit_comments_path: str,
    sample: int = 1000,
    seed: int = 10152024,
) -> Tuple[DataFrame, DataFrame]:
    """Build the wiki articles and reddit comments dataset and apply preprocessing to wiki article text.

    Args:
        wiki_articles_path (str): Path to the wiki articles in the data directory.
        reddit_comments_path (str): Path to the reddit comments in the data directory.
        sample (int, optional): Size of the sample. Defaults to 1000.
        seed (int, optional): Seed for reproducible results. Defaults to 10152024.

    Returns:
        Tuple[DataFrame, DataFrame]: wiki_articles, reddit
    """
    wiki_articles = pd.read_parquet(wiki_articles_path)
    reddit = pd.read_csv(reddit_comments_path)

    wiki_articles = wiki_articles.sample(n=sample, random_state=seed)
    reddit = reddit.sample(n=sample, random_state=seed)
    wiki_articles["text_tokens"] = wiki_articles["text"].apply(preprocess_articles)

    wiki_articles = wiki_articles[["title", "text", "text_tokens"]]

    reddit.rename(
        columns={"comment": "text", "comment_tokens": "text_tokens"}, inplace=True
    )

    reddit = reddit[["post_title", "subreddit", "text", "text_tokens"]]

    return wiki_articles, reddit
