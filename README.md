# Collaborative Filtering: Replication & Cross-Domain Study

Replication and extension of **Bangaru & Kumar, *Collaborative Filtering Models: An Experimental and Detailed Comparative Study*** (Scientific Reports, 2025). We reproduce one of the paper's central claims — that traditional methods (KNN, SVD) are strong for *rating prediction*, while neural and graph-based models are better for *top-k recommendation* — and extend it with a stricter evaluation protocol, an extra baseline, additional metrics, hyperparameter sensitivity analysis, and a cross-domain test on music data.

**Authors:** Muhil Ramesh, Tejas Hegde
**Date:** May 2026

---

## Models

| Model | Category | Implementation |
|-------|----------|----------------|
| KNNBasic (Item-Item) | Memory-based CF | Surprise |
| SVD | Matrix factorization | Surprise |
| NCF | Neural CF | PyTorch (from scratch) |
| LightGCN | Graph-based CF | PyTorch (from scratch, BPR loss) |
| Popularity | Non-personalized baseline | Custom |

## Datasets

| Dataset | Interactions | Users | Items | Domain |
|---------|-------------:|------:|------:|--------|
| MovieLens 1M | 1,000,209 | 6,040 | 3,706 | Movies |
| Last.fm HetRec 2011 | 71,355 | 1,859 | 2,823 | Music |

Last.fm provides user-artist listening counts rather than explicit ratings; these were converted into pseudo-ratings via log scaling so the same full-ranking evaluation could be applied.

## Evaluation

Two protocols are implemented:

- **Observed-test-pair** — models ranked only over items in the test set (useful as a sanity check, but inflates scores).
- **Full-catalog ranking** — for each user, all unseen items are ranked, training items removed, and the top 10 selected. This is the stricter, more realistic setting and the basis for our main conclusions.

Metrics: **RMSE, MAE, Precision@10, Recall@10, F1@10, NDCG@10, Coverage**.

---

## Key Results

**MovieLens 1M — full-catalog ranking (NDCG@10):**

| Model | Precision@10 | Recall@10 | F1@10 | NDCG@10 |
|-------|-------------:|----------:|------:|--------:|
| LightGCN | 0.2387 | 0.1605 | 0.1920 | **0.2911** |
| NCF | 0.1693 | 0.1097 | 0.1331 | 0.1958 |
| Popularity | 0.1520 | 0.0907 | 0.1136 | 0.1767 |
| SVD | 0.0781 | 0.0433 | 0.0557 | 0.0917 |
| KNNBasic | 0.0313 | 0.0074 | 0.0120 | 0.0260 |

**Cross-domain (NDCG@10):**

| Model | MovieLens 1M | Last.fm |
|-------|-------------:|--------:|
| LightGCN | 0.2911 | 0.0615 |
| NCF | 0.1958 | 0.0117 |
| Popularity | 0.1767 | 0.0266 |
| SVD | 0.0917 | 0.0036 |
| KNNBasic | 0.0260 | 0.0033 |

**Takeaways**

- **LightGCN is the strongest top-k recommender** on both datasets, followed by NCF — supporting the original paper's main conclusion.
- For pure **rating prediction**, SVD (RMSE 0.8736) beats KNNBasic (RMSE 0.9993), but both fall apart under full-catalog ranking.
- **Evaluation protocol changes the conclusions.** Under observed-test-pair evaluation SVD looked competitive; under full-catalog ranking the neural/graph models clearly win.
- **Popularity is a deceptively strong baseline** — on Last.fm it outranks NCF, showing popularity effects dominate some domains.
- **Bigger isn't always better:** KNN peaks at *k* = 40 and SVD at ~50 latent factors, with performance degrading beyond that.

---

## Extensions Beyond the Original Paper

- Full-catalog ranking evaluation protocol
- Non-personalized Popularity baseline
- Coverage as an additional metric
- KNN neighborhood-size and SVD latent-factor sensitivity analysis
- NCF and LightGCN re-implemented in PyTorch
- Cross-domain evaluation on Last.fm HetRec 2011

---

## Repository Contents

```
.
├── ReplicationFinal (2).ipynb   # Main notebook (all models, evaluation, plots)
├── Replication_Paper (1).pdf           # Full write-up with results and discussion
└── README.md
```

## Running the Code

The project runs as a single notebook on **Google Colab**:

1. Open `collaborative_filtering.ipynb` in Google Colab.
2. (Optional) Enable a GPU runtime for faster NCF/LightGCN training: *Runtime → Change runtime type → GPU*.
3. Run all cells top to bottom. Dependency installs (`scikit-surprise`, `torch`, etc.) are handled in the first cells.
4. Make the datasets available before running the evaluation cells:
   - [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
   - [Last.fm HetRec 2011](https://grouplens.org/datasets/hetrec-2011/)

   Either upload them to the Colab session or mount Google Drive and update the dataset paths at the top of the notebook.

**Main libraries:** PyTorch, scikit-surprise, NumPy, Pandas, Matplotlib.

---

## Reference

Rajesh Devangam Bangaru and Avadhesh Kumar. *Collaborative Filtering Models: An Experimental and Detailed Comparative Study.* Scientific Reports, vol. 15, no. 1, 2025. https://doi.org/10.1038/s41598-025-15096-4
