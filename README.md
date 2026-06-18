# Link Prediction in Protein–Protein Interaction Networks
### Stacking Methods via Supervised Rank Aggregation

**Master's Thesis — LIP6, Sorbonne Université / CNAM Paris**  
Supervised by [Prof. Lionel Tabourier](https://lip6.fr) · Complex Networks Team, LIP6  
Defended: October 2025

---

## Overview

Protein–protein interaction (PPI) networks are fundamentally incomplete, less than 10% of human
protein interactions have been experimentally confirmed. This project builds a **reproducible
pipeline** to predict missing links in PPI networks by combining multiple graph-theoretic scoring
methods through supervised rank aggregation.

The core question: can we improve over the best individual predictor by learning how to
*merge* complementary rankings in a data-driven way?

---

## Methods Implemented

### Unsupervised Topological Heuristics (weak learners)

| Method | Family | Key Idea |
|--------|--------|----------|
| **CN** — Common Neighbors | Neighborhood overlap | Two proteins are likely to interact if they share many direct neighbors |
| **AA** — Adamic-Adar | Neighborhood overlap | Shared neighbors with low degree carry more signal |
| **CRA** — Common Resource Allocation | Local community | Weights shared neighbors by their local cluster density |
| **L3** | Path-based | Proteins connected by many length-3 paths are more likely to interact (Kovács et al.) |
| **L3N** | Path-based | Normalized L3 — rewards P4 evidence, penalizes contradictory P3 patterns, reduces hub bias (Yuen & Jansson) |

**Why L3 and not Common Neighbors?** In PPI networks, proteins interact because they are
*complementary* (like puzzle pieces), not because they are *similar*. L3 captures this
compatibility signal; CN does not.

### Supervised Rank Aggregation

**RankMerging** (Tabourier et al., 2019), a learning-to-rank framework that merges several
unsupervised rankings into a single, more accurate ordering.

- Each weak learner produces a ranked list of candidate protein pairs
- A sliding window of size `g` estimates the local hit rate of each ranking
- The method learns *where* each heuristic is most reliable along the list
- At test time, the learned mixing profile is replayed  **no labels required**
- Time complexity: O(α · θ), linear in the number of input rankings and output length

---

## Datasets

| Dataset | Description | Scale |
|---------|-------------|-------|
| **Hein et al.** | Human co-complex interactome (AP–MS, HeLa cells) | Small, well-curated |
| **STRING (human)** | Large-scale human PPI network, experimentally validated "binding" edges only | Large, noisy |

Data splitting follows two strategies:
- **Yuen split** — 25% train / 25% validation / 50% test
- **Natural split** — 50% train / 25% validation / 25% test

---

## Key Results

Evaluated with **Precision–Recall curves** and **AUPRC** (Area Under Precision-Recall Curve),
which is appropriate for highly imbalanced link prediction tasks.

**On the Hein dataset (Yuen split, g=3000):**

| Method | AUPRC |
|--------|-------|
| L3N | ~0.175 |
| RankMerging | ~0.170 |
| CRA | ~0.053 |
| CN | ~0.049 |
| AA | ~0.053 |

**On STRING (human) — RankMerging outperforms all individual weak learners** (AUPRC ~0.626 vs.
L3N ~0.621), with consistent gains across splits and window sizes.

**Main finding:** L3N is the strongest single scorer on both datasets. RankMerging closely tracks
or slightly surpasses it at the top of the ranking, while consistently outperforming CN/AA/CRA.
The bottleneck is L3N scoring (path enumeration); the rank aggregation step itself is fast.

---

## Repository Structure

```
karaneh_rankmerging/
├── main.py               # Entry point: runs scoring, rank aggregation, and plotting
├── methods.py            # Scoring methods: CN, AA, CRA, L3, L3N
├── data_processing.py    # Graph loading, edge list construction, data splitting
├── utils.py              # TP/FP computation, PR curve utilities, plotting
├── config.py             # Dataset paths and experiment configuration
├── dataset/              # PPI datasets (Hein et al., STRING human)
├── scores/               # Precomputed scores per method per dataset
├── potential_pairs/      # Candidate non-edges at distance 3
└── result/plots/         # Generated PR curves and contribution plots
```

---

## How to Run

**Requirements:** Python 3.9+, NumPy, Pandas, Matplotlib, Scikit-learn, NetworkX

```bash
git clone https://github.com/karanehzm/karaneh_rankmerging.git
cd karaneh_rankmerging
pip install numpy pandas matplotlib scikit-learn networkx
```

**Configure your experiment** in `config.py` , set `DATASET_NAME`, data paths, and directories.

**Run the pipeline:**

```bash
python main.py
```

This will:
1. Load the dataset and compute scores (CRA, L3N) on candidate pairs
2. Run the RankMerging learning phase (requires the external `Merge_learn` binary from Tabourier et al.)
3. Run the RankMerging testing phase (`Merge_test`)
4. Plot combined Precision–Recall curves for all methods

**RankMerging binary commands (examples):**

```bash
# Hein dataset
./Merge_learn 102800318 100000 300 2 l3_learning_heinetal-rec.txt,cra_learning_heinetal-rec.txt ex_100k_heinetal-rec
./Merge_test 102800318 100000 learning_ex_100k_heinetal-rec.txt 1.5 2 l3_testing_heinetal-rec.txt,cra_testing_heinetal-rec.txt ex_t_100k_heinetal-rec

# STRING (human)
./Merge_learn 485664 100000 300 2 l3_learning_human_ppi_lcqb_s900.txt,cra_learning_human_ppi_lcqb_s900.txt ex_100k_human_ppi_lcqb_s900
./Merge_test 485664 100000 learning_ex_100k_human_ppi_lcqb_s900.txt 1.5 2 l3_testing_human_ppi_lcqb_s900.txt,cra_testing_human_ppi_lcqb_s900.txt ex_t_100k_human_ppi_lcqb_s900
```

---

## References

- Kovács et al., *Network-based prediction of protein interactions*, Nature Communications, 2019
- Yuen & Jansson, *Normalized L3-based link prediction in PPI networks*, BMC Bioinformatics, 2023
- Tabourier et al., *RankMerging: a supervised learning-to-rank framework*, Machine Learning, 2019
- Cannistraci et al., *From link-prediction in brain connectomes to the local-community-paradigm*, Scientific Reports, 2013

---

## Author

**Karaneh Zolfaghari Moghaddam**  
M.Sc. Computer Networks & IoT — CNAM Paris / Sorbonne Université  
French Government Eiffel Excellence Scholar (2023)  
[github.com/karanehzm](https://github.com/karanehzm) · karanehzolfaghari@gmail.com
