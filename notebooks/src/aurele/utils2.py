# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    utils2.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aurele <aurele@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/02/20 15:14:48 by aurele            #+#    #+#              #
#    Updated: 2026/02/21 20:49:06 by aurele           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import urllib.request
from rank_bm25 import BM25Okapi

def ensure_data(dir, file, url):
    if not file.exists():
        print(f"Data not found at {file}. Dowloading from GitHub")
        dir.mkdir(parents = True, exist_ok = True)

        try:
            urllib.request.urlretrieve(url, file)
            print("Download successful!")
        except Exception as e:
            print(f"Failed to download adata: {e}")
    else: 
        print("Data found locally, skipping download")

def load_reviews(csv_path, lang="en"):
    """
    Default language is English
    We just kept reviews, note and idplace
    we droped reviews with no text
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df['langue'] == lang]
    df = df[["idplace", "review", "note"]]
    df = df.dropna(subset=["review"])
    return df


def ft_clean_text(text):
    """
    Cleaning steps:
    - lowercase
    - remove punctuation
    """
    text = text.lower()
    text = re.sub(r"[^a-z.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_reviews(df):
    """Apply text cleaning to all reviews."""
    df = df.copy()
    df["review"] = df["review"].apply(ft_clean_text)
    return df


def plot_reviews_distribution(reviews, id_col="idplace"):
    counts = reviews[id_col].value_counts()
    counts = counts[counts > 0]
    plt.figure(figsize=(10, 4))
    plt.plot(counts.to_numpy())
    plt.yscale("log")
    plt.xlabel("Restaurants (ranked)")
    plt.ylabel("Number of reviews (log scale)")
    plt.title("Distribution of reviews per restaurant")
    plt.show()


def aggregate_by_place(df):
    """
    Group reviews by idplace:
    - concatenate all reviews into a single text
    - compute the average note
    """
    grouped_df = (
        df
        .groupby("idplace")
        .agg({
            "review": lambda x: ".".join(x),
            "note": "mean"
        })
        .reset_index()
    )
    return grouped_df


# ====================================
# TF-IDF

def compute_tfidf(reviews_series, max_features=5000):
    """max_features: most frequent terms cap"""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X_tfidf = vectorizer.fit_transform(reviews_series)
    return X_tfidf, vectorizer


def filter_sparse_vectors(X_tfidf, idplaces, zero_ratio_thresh=0.40):
    """
    Drop rows where more than zero_ratio_thresh of features are zero.
    Returns filtered matrix + filtered idplaces.
    """
    n_features = X_tfidf.shape[1]
    non_zero   = (X_tfidf > 0).sum(axis=1).A1
    zero_ratio = 1 - non_zero / n_features

    kept_mask = zero_ratio <= zero_ratio_thresh
    print(f"[filter] dropped {(~kept_mask).sum()} / {X_tfidf.shape[0]} docs "
          f"(>{zero_ratio_thresh * 100:.0f}% zeros)")
    return X_tfidf[kept_mask], idplaces[kept_mask]


# ====================================
# Similarity index & rankings

def build_topk_similarity_index(cos_sim, idplaces, k=5):
    """For each place, store its top-k neighbours and scores."""
    rows = []
    for i, pid in enumerate(idplaces):
        sims  = cos_sim[i]
        order = np.argsort(sims)[::-1]
        order = order[order != i][:k]
        row   = {"idplace": pid}
        for j, idx in enumerate(order):
            row[f"top{j+1}_id"]  = idplaces[idx]
            row[f"top{j+1}_sim"] = sims[idx]
        rows.append(row)
    return pd.DataFrame(rows)


def rank_places_by_mean_similarity(cos_sim, idplaces):
    """Rank places by avg cosine similarity to all others."""
    N        = cos_sim.shape[0]
    mean_sim = (cos_sim.sum(axis=1) - 1) / (N - 1)
    return (pd.DataFrame({"idplace": idplaces, "mean_cosine_similarity": mean_sim})
              .sort_values("mean_cosine_similarity", ascending=False)
              .reset_index(drop=True))


def rank_places_by_topk_similarity(cos_sim, idplaces, k=5):
    """Rank places by avg similarity to their top-k neighbours."""
    scores = []
    for i in range(len(idplaces)):
        sims  = cos_sim[i]
        order = np.argsort(sims)[::-1]
        order = order[order != i][:k]
        scores.append(sims[order].mean())
    return (pd.DataFrame({"idplace": idplaces, f"mean_top{k}_similarity": scores})
              .sort_values(f"mean_top{k}_similarity", ascending=False)
              .reset_index(drop=True))


def count_topk_appearances(cos_sim, idplaces, k=10):
    """Count how often each place appears in others' top-k lists."""
    counts = {pid: 0 for pid in idplaces}
    for i in range(len(idplaces)):
        sims  = cos_sim[i]
        order = np.argsort(sims)[::-1]
        order = order[order != i][:k]
        for idx in order:
            counts[idplaces[idx]] += 1
    return (pd.DataFrame({"idplace": list(counts.keys()),
                          f"count_in_top{k}": list(counts.values())})
              .sort_values(f"count_in_top{k}", ascending=False)
              .reset_index(drop=True))


# ====================================
# Export

import json

def export_similarity_to_json(similarity_index, output_path="similarity_results.json"):
    """
    Write the top-k similarity index to a JSON file.
    Each entry: idplace + list of {id, confidence} neighbours.
    """
    results = []
    for _, row in similarity_index.iterrows():
        neighbours = []
        j = 1
        while f"top{j}_id" in row:
            neighbours.append({
                "id":         int(row[f"top{j}_id"]),
                "confidence": round(float(row[f"top{j}_sim"]), 4)
            })
            j += 1
        results.append({
            "idplace":    int(row["idplace"]),
            "neighbours": neighbours
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[export] {len(results)} places written to {output_path}")

def safe_export_similarity(similarity_index, output_path):
    
    # Check if the file already exists
    if output_path.exists():
        user_choice = input(f"File '{output_path.name}' already exists. Overwrite? (y/n): ").lower()
        
        if user_choice != 'y':
            print("Export cancelled.")
            return
    
    # Call your original function if file doesn't exist or user said 'y'
    export_similarity_to_json(similarity_index, output_path=output_path)
    print(f"Successfully exported to {output_path}")

# ====================================
# Inspection / tests

def test_tfidf_vocab(X_tfidf, vectorizer, n_words=8, n_docs=4):
    """
    Compare word distributions across documents.
    Selects words that have many zeros overall but not at the same places
    → shows meaningful variance in the vocabulary.
    """
    vocab   = np.array(vectorizer.get_feature_names_out())
    X_dense = np.asarray(X_tfidf.todense())

    # words present in 10-60% of docs → interesting, not too rare/common
    nonzero_ratio = (X_dense > 0).mean(axis=0)
    candidates    = np.where((nonzero_ratio > 0.10) & (nonzero_ratio < 0.60))[0]

    # keep the most discriminating ones (highest std)
    stds     = X_dense[:, candidates].std(axis=0)
    top_idx  = candidates[np.argsort(stds)[::-1][:n_words]]
    selected = vocab[top_idx]

    print(f"Selected words: {list(selected)}\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # bar chart: how often each word appears across docs
    presence = (X_dense[:, top_idx] > 0).mean(axis=0)
    axes[0].bar(selected, presence)
    axes[0].set_title("Fraction of docs containing each word")
    axes[0].set_ylabel("Presence ratio")
    axes[0].tick_params(axis="x", rotation=40)

    # heatmap: actual TF-IDF scores for the first n_docs docs
    sample = X_dense[:n_docs, :][:, top_idx]
    im = axes[1].imshow(sample, aspect="auto", cmap="YlOrRd")
    axes[1].set_xticks(range(len(selected)))
    axes[1].set_xticklabels(selected, rotation=40)
    axes[1].set_yticks(range(n_docs))
    axes[1].set_yticklabels([f"doc {i}" for i in range(n_docs)])
    axes[1].set_title(f"TF-IDF scores — first {n_docs} docs")
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.show()

    summary = pd.DataFrame(sample, columns=selected,
                           index=[f"doc {i}" for i in range(n_docs)])
    print(summary.round(3))

# =========================================
# BM25 model

def get_important_vocab(reviews_series, max_features=300):
    """Identifie les mots les plus importants du corpus."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    vectorizer.fit(reviews_series)
    return set(vectorizer.get_feature_names_out())

def compute_bm25_fast(reviews_series, idplaces, k=5, max_vocab = 80):
    """
    Computes Top-K recommendation for each place using BM25.

    Parameters:
        reviews_series: pd.Series containing aggregated & clean reviews
        idplaces: list/array of ids corresponding to each review
        k: number of desired recommendations

    Returns:
        DataFrame containing for each idplace the k-neighbours and their BM25 scores
    """
    vocab = get_important_vocab(reviews_series, max_features=max_vocab)
    
    print(f"Filtrage des tokens (Vocabulaire limité à {max_vocab} mots)...")
    tokenized_corpus = []
    for doc in reviews_series:
        words = str(doc).replace('.', ' ').split()
        filtered_words = [w for w in words if w in vocab]
        tokenized_corpus.append(filtered_words)

    print("Initialisation BM25...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("Calcul des recommandations (Vectorisé)...")
    rows = []
    for i, pid in enumerate(idplaces):
        query = tokenized_corpus[i]
        if not query: continue
            
        scores = bm25.get_scores(query)
        scores[i] = -1  
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        
        
        row = {"idplace": pid}
        for j, idx in enumerate(top_k_idx):
            row[f"top{j+1}_id"] = idplaces[idx]
            row[f"top{j+1}_sim"] = round(float(scores[idx]), 4)
        rows.append(row)

    return pd.DataFrame(rows)