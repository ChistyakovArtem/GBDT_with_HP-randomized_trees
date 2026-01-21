# Faster and Stronger CatBoost
## Implicit Ensembling via Hyperparameter-Randomized Trees

This repository explores how introducing **hyperparameter diversity at the level of individual trees** inside a single GBDT training loop can recover much of the benefit of Kaggle-style hyperparameter optimization (HPO) and ensembling — while keeping a single training run, fewer trees, and significantly lower computational cost.

I study two closely related ideas:
- randomizing hyperparameters for each tree,
- explicitly optimizing hyperparameters of each tree using a lightweight Optuna loop.

Both approaches aim to replace expensive global HPO and explicit ensembling with **online, per-tree diversity**.

---

## Motivation

Consider a realistic large-scale setup where a single `fit → predict` cycle takes **X = 5 minutes**.

In Kaggle-style workflows, strong performance is typically achieved using:
- hyperparameter optimization (often ~50 full trainings),
- ensembling (often ~3 models).

Even with conservative assumptions, this increases total training time by roughly **150×**.

In real-world research and production systems (e.g., recommender systems, multi-target modeling), this is often infeasible:
- experiments that used to take a weekend now take months,
- pipelines require many data splits,
- validation data is reused multiple times,
- overfitting risk increases,
- code complexity grows substantially.

Outside leaderboard-driven benchmarks, this approach is usually overkill.

---

## Core Idea

### One conceptual change to GBDT training

**Before**  
Each new tree is trained using a fixed hyperparameter configuration chosen once by the user.

**After**  
Each new tree is trained using a **different hyperparameter configuration**, sampled or optimized online.

As a result, a single boosting run behaves like an **implicit ensemble of heterogeneous base learners**, without multiple training runs or explicit ensembling.

---

## Model 1 — Randomized Hyperparameters per Tree
`problem_1_train_speed`

In the first model, hyperparameters of each tree are **randomly sampled**.

This introduces structural diversity between trees and implicitly mixes different inductive biases inside a single boosting process. The final model behaves similarly to an ensemble of models trained under different hyperparameter regimes, but is trained in one pass.

### Empirical Effects

- Early stopping is reached **40–50% faster**.
- Training and inference are roughly **2× faster**.
- Significantly **fewer trees** are required.
- Predictive quality is competitive with, and often superior to, standard CatBoost.

---

### Experimental Setup (Model 1)

#### Baselines

- **CatBoost**  
  Standard CatBoost with default configuration:  
  `n_estimators = 2000`, `learning_rate = 0.1`

- **MyCatBoost**  
  Custom boosting implementation using one-tree CatBoost models:  
  `n_estimators = 1`, `learning_rate = 1`

#### Proposed Method

- Same boosting logic as MyCatBoost
- Per-tree hyperparameters are randomly sampled.
- Implementation in Python using one-tree CatBoost models as base learners.

---

### Results Across 13 Datasets (Model 1)

**Accuracy**
- 6 / 13 datasets outperform standard CatBoost
- 11 / 13 datasets outperform MyCatBoost

**Efficiency (number of trees)**
- 13 / 13 datasets use fewer trees than CatBoost
- 9 / 13 datasets use fewer trees than MyCatBoost

Results file: `problem_1_train_speed/benchmark_results.csv`

---

## Model 2 — Per-Tree Optuna Optimization
`problem_2_test_quality`

This model explicitly optimizes hyperparameters for each new base learner using Optuna on a small hold-out set (`HPO_set`):

1. Train multiple candidate trees on training data.
2. Evaluate each candidate on `HPO_set`.
3. Select the **best-performing tree** for inclusion in the ensemble.

Effectively, each new tree is the result of a local HPO search, dynamically chosen at each boosting iteration.

### Preliminary Results (Model 2)

- Average number of trees **reduced ~8×** versus standard CatBoost.
- Accuracy **matches or exceeds** CatBoost with full global HPO.

**Note:** Experiments are **not fully completed** due to computational limits. Full benchmarks require a dedicated server. Current results are strong empirical evidence, not final.

---

## Why This Works

Both approaches leverage **tree-level diversity**:

- Improves hypothesis space coverage.
- Mimics explicit ensembling or large-scale HPO.
- Achieves a strong bias–variance trade-off with **a single training run**.

Advantages:
- Avoid repeated dataset passes.
- Avoid multiple model checkpoints.
- Avoid complex orchestration logic.

---

## Takeaways

- Hyperparameter-randomized trees can deliver **Kaggle-level performance** in a fraction of the time.
- Single training run with minimal code.
- Strong regularization and practical feasibility for large-scale real-world systems.
- A pragmatic alternative when **speed, simplicity, and robustness** matter more than marginal leaderboard gains.

