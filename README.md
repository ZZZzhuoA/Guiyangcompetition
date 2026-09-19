# Competition Strategy Recommendation Experiments

This project contains staged experiments for the counter-UAV strategy recommendation task.

## Stage 1

Goal: build a reproducible baseline without external package installation.

Implemented methods:

- `majority_strategy`: always recommends the most frequent training strategy.
- `strategy_prototype`: assigns a sample to the nearest strategy centroid.
- `knn`: recommends from similar historical scenarios.
- `outcome_knn`: estimates the utility of each candidate strategy from similar historical samples under that strategy, then chooses the best estimated utility.

Outputs are written to `outputs/stage1/`.

Run:

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" experiments/stage1_baselines.py
```

## Proposed Later Stages

- Stage 2: stronger supervised models and engineered features.
- Stage 3: adaptive strategy generation with scenario clustering and outcome-aware ranking.
- Stage 4: final report, ablation study, and test-set submission packaging.

## Additional ML Comparison

Run:

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" experiments/stage4_more_ml.py
```

This compares ridge regression, strategy-wise ridge regression, RBF kernel regression, extremely randomized trees, and gradient boosted stumps under the required equal-weight utility rule.

## Dual Metric Regression

Run:

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" experiments/stage5_dual_metric.py
```

This trains separate regressors for interception rate and cost-effectiveness, then combines the two predictions with equal weights.

## Pairwise Ranking

Run:

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" experiments/stage6_pairwise_ranking.py
```

This compares strategy-pair ranking methods that learn whether one strategy beats another under the same scenario.

## Pareto Multi-Objective Recommendation

Run:

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" experiments/stage7_pareto_multiobjective.py
```

This predicts interception rate and cost-effectiveness separately, filters dominated strategies with Pareto dominance, and applies tie-break rules for final recommendation.

## Dual-View Clustering

Run:

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" experiments/stage9_dual_view_clustering.py
```

This clusters resource-state features and threat-situation features separately, then recommends strategies from combined resource-threat scenarios.

## Fusion Recommendation

Run:

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" experiments/stage10_fusion.py
```

This fuses scenario-local utility, pairwise ranking, Pareto dominance, and dual-view clustering scores.

## Feature Engineering

Run:

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" experiments/stage11_feature_engineering.py
```

This adds threat intensity, resource capability, and offense-defense pressure features, then compares raw and engineered feature sets.
