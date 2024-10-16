import itertools
from typing import Dict, List
from collections import Counter
import numpy as np
from pandas import DataFrame

from nltk.util import ngrams


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """Calculates the Kullback-Leibler (KL) divergence between two probability distributions P and Q.

    D_{KL}(P \parallel Q) = \sum_{x} P(x) \log\left(\frac{P(x)}{Q(x)}\right)

    Example usage:
        p = np.array([0.4, 0.6])
        q = np.array([0.5, 0.5])

        >>> kl_div(p, q)
        np.float64(0.020135513550688863)

    Args:
        p (np.ndarray): The baseline probability distribution P
        q (np.ndarray): The approximate probability distribution Q

    Returns:
        float:  The KL Divergence D(P || Q)
    """
    q = np.where(q > 0, q, 1e-10)
    return np.sum(p * np.log(p / q))


def build_unigram_model(tokens: List) -> Counter:
    """Builds a unigram model from a list of word tokens.

    Counts the occurrences of each word in the tokenized text and represents the relative frequency of each word.

    Args:
        tokens (List): List of tokens from the text.

    Returns:
        Counter: Values are the relative frequencies of those tokens.
    """
    unigram_model = Counter(tokens)
    total_count = sum(unigram_model.values())
    for word in unigram_model:
        unigram_model[word] /= total_count

    return unigram_model


def build_bigram_model(tokens: List) -> Counter:
    """Builds a bigram model from a list of tokens.

    Generates bigrams from the tokenized text, counts their occurrences, and then normalizes the counts to represent the relative frequency of each bigram.

    Args:
        tokens (List): List of tokens from the text.

    Returns:
        Counter: Values are the relative frequencies of the bigrams.
    """
    bigrams = list(ngrams(tokens, 2))
    bigram_model = Counter(bigrams)
    total_count = sum(bigram_model.values())
    for bigram in bigram_model:
        bigram_model[bigram] /= total_count
    return bigram_model


def build_trigram_model(tokens: List) -> Counter:
    """Builds a trigram model from a list of tokens.

    Generates trigrams from the tokenized text, counts their occurrences, and then normalizes the counts to represent the relative frequency of each trigram.

    Args:
        tokens (List): List of tokens from the text.

    Returns:
        Counter: Values are the relative frequencies of the bigrams.
    """
    trigrams = list(ngrams(tokens, 3))
    trigram_model = Counter(trigrams)
    total_count = sum(trigram_model.values())
    for trigram in trigram_model:
        trigram_model[trigram] /= total_count
    return trigram_model


def calc_kl_dataset(dataset: DataFrame, models_dict: Dict) -> Dict:
    """Calculate the KL divergence for all datasets.

    Args:
        dataset (DataFrame): Corpora
        models_dict (Dict): Dictionary of models

    Returns:
        dict: All KL divergences calculated.
    """
    all_kl_divergences = {}

    for idx, tokens in enumerate(dataset["text_tokens"]):

        models = {
            "unigram": models_dict["unigram"](tokens),
            "bigram": models_dict["bigram"](tokens),
            "trigram": models_dict["trigram"](tokens),
        }

        distributions = {}
        kl_divergences = {}

        for m1, m2 in itertools.combinations(["unigram", "bigram", "trigram"], r=2):
            all_keys = set(models[m1].keys()).union(set(models[m2].keys()))
            p_dist = [models[m1].get(k, 1e-10) for k in all_keys]
            q_dist = [models[m2].get(k, 1e-10) for k in all_keys]

            distributions[(m1, m2)] = (p_dist, q_dist)

        for key in distributions.keys():
            p = np.array(distributions[key][0])
            q = np.array(distributions[key][1])
            kl_divergences[key] = kl_div(p, q)

        all_kl_divergences[idx] = kl_divergences

    return all_kl_divergences
