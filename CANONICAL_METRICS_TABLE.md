# Canonical Metrics Table

| File | Model P@50 | Baseline P@50 | Base Rate | Status |
|------|------------|---------------|-----------|--------|
| work/capstone_report.md | 0.780 | 0.640 | 0.542 | ✅ Aligned |
| docs/paper/index.html | 0.780 | 0.640 | 0.542 | ✅ Aligned |
| work/notebooks/capstone.ipynb | 0.780 | 0.640 | 0.542 | ✅ Aligned |
| work/notebooks/w01_research_question.ipynb | 0.780 | 0.640 | 0.542 | ✅ Aligned |
| work/notebooks/w04_baseline_score.ipynb | 0.780 | 0.640 | 0.542 | ✅ Aligned |
| work/outputs/canonical_metrics.json | 0.780 | 0.640 | 0.542 | ✅ Receipts created |

**Canonical Results Summary:**
- Base Rate: 0.542 (54.2% declining rate in dataset)
- Fair Baseline Precision@50: 0.640 (NO label leakage)
- Random Forest Model Precision@50: 0.780
- Model improvement over baseline: 1.22×
- Model improvement over random: 1.44×

**Feature Importance (Random Forest, 5 features):**
1. impressions_90d: 0.417
2. avg_position: 0.237
3. content_age_days: 0.204
4. ctr: 0.079
5. days_since_last_update: 0.063

**Validation Configuration:**
- Method: Client-holdout (GroupKFold, n_splits=5)
- Random seed: 42
- Train size: 22,992 pages
- Test size: 7,008 pages

**Charts:**
- comparison.png: Model vs Fair Baseline Precision@50
- feature_importance.png: Feature Importance (5 features)
- error_analysis.png: Error Analysis in Top 50 Predictions

All numbers are consistent across all artifacts with NO label leakage.