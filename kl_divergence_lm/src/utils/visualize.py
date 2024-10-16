from typing import Dict
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])


def plot_kl_divergences(kl_divergences: Dict, dataset_name: str) -> None:
    """Plot the KL Divergences for a given dataset.

    Args:
        kl_divergences (dict): Dictionary containing the KL divergences for each entry.
        dataset_name (str): Name of the dataset (used in the plot title).
    """

    indices = list(kl_divergences.keys())
    kl_unigram_bigram = [
        kl_divergences[idx].get(("unigram", "bigram"), 0) for idx in indices
    ]
    kl_unigram_trigram = [
        kl_divergences[idx].get(("unigram", "trigram"), 0) for idx in indices
    ]
    kl_bigram_trigram = [
        kl_divergences[idx].get(("bigram", "trigram"), 0) for idx in indices
    ]

    plt.figure(figsize=(10, 8))
    plt.plot(indices, kl_unigram_bigram, label="Unigram vs Bigram", marker="o")
    plt.plot(indices, kl_unigram_trigram, label="Unigram vs Trigram", marker="x")
    plt.plot(indices, kl_bigram_trigram, label="Bigram vs Trigram", marker="s")

    plt.xlabel("Index")
    plt.ylabel("KL Divergence")
    plt.title(f"KL Divergence for {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.show()
