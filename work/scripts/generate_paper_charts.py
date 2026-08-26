"""Rebuild docs/paper/charts from the warehouse frame and canonical_metrics.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from warehouse_frame import FEATURES, load_page_frame, precision_at_k, repo_root

CHARTS = ROOT / "docs" / "paper" / "charts"
RULE = "#c44e52"
FOREST = "#4c72b0"
BASE = "#8c8c8c"
MEAN = "#55a868"


def _save(fig, name: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    path = CHARTS / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def _metrics() -> dict:
    path = ROOT / "work" / "outputs" / "canonical_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fold1(df):
    X = df[FEATURES].fillna(0)
    y = df["is_declining"]
    train_idx, test_idx = next(GroupKFold(n_splits=5).split(X, y, df["client_id"]))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_df = X_test.copy()
    test_age = df.iloc[test_idx]["content_age_days"].fillna(0).values
    stale = ((test_df["days_since_last_update"] >= 180) & (test_df["impressions_90d"] >= 500)).astype(int)
    pos = ((test_df["avg_position"] > 10) & (test_age >= 180)).astype(int)
    low = ((test_df["ctr"] < 0.05) & (test_df["impressions_90d"] >= 1000)).astype(int)
    rule = (
        0.4 * stale * test_df["impressions_90d"]
        + 0.3 * pos * test_df["impressions_90d"]
        + 0.3 * low * test_df["impressions_90d"]
    ).to_numpy()
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    forest = rf.predict_proba(X_test)[:, 1]
    return forest, rule, y_test.to_numpy()


def precision_curve(scores, labels, ks):
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order]
    return [float(y[:k].mean()) for k in ks]


def plot_performance(m: dict) -> None:
    labels = [
        "Fair rule\n(fold 1)",
        "Forest\n(5-fold mean)",
        "Test-fold\nbase rate",
        "Forest\n(fold 1)",
    ]
    vals = [
        m["fair_baseline_precision_at_50"],
        m["five_fold_mean_precision_at_50"],
        m["base_rate"],
        m["random_forest_precision_at_50"],
    ]
    colors = [RULE, MEAN, BASE, FOREST]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Precision@50")
    ax.set_ylim(0, 1.0)
    ax.set_title("Precision@50 on Hugging Face warehouse (client-holdout)")
    ax.axhline(m["base_rate"], color=BASE, linestyle="--", linewidth=1, alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "performance.png")


def plot_comparison(m: dict) -> None:
    labels = ["Fair rule", "Forest (fold 1)", "Test-fold base rate"]
    vals = [
        m["fair_baseline_precision_at_50"],
        m["random_forest_precision_at_50"],
        m["base_rate"],
    ]
    colors = [RULE, FOREST, BASE]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Precision@50")
    ax.set_ylim(0, 1.0)
    ax.set_title("Model vs fair baseline Precision@50 (fold 1)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "comparison.png")


def plot_importances(m: dict) -> None:
    imp = m["feature_importance"]
    names = sorted(imp, key=imp.get, reverse=True)
    vals = [imp[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(names[::-1], vals[::-1], color=FOREST, edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Feature importance")
    ax.set_title("Random Forest importances (warehouse fold 1)")
    ax.set_xlim(0, 0.55)
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "top15.png")
    fig2, ax2 = plt.subplots(figsize=(8, 4.2))
    ax2.barh(names[::-1], vals[::-1], color=FOREST, edgecolor="black", linewidth=0.6)
    ax2.set_xlabel("Feature importance")
    ax2.set_title("Feature importance (Random Forest, 5 features)")
    ax2.set_xlim(0, 0.55)
    for y, v in enumerate(vals[::-1]):
        ax2.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    _save(fig2, "feature_importance.png")
    fig3, ax3 = plt.subplots(figsize=(8, 4.2))
    ax3.barh(names[::-1], vals[::-1], color=FOREST, edgecolor="black", linewidth=0.6)
    ax3.set_xlabel("Importance")
    ax3.set_ylabel("feature")
    ax3.set_title("Top feature importances (warehouse fold 1)")
    ax3.set_xlim(0, 0.55)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    fig3.tight_layout()
    _save(fig3, "top.png")


def plot_precision_at_k(forest, rule, y) -> None:
    ks = list(range(10, 201, 10))
    forest_curve = precision_curve(forest, y, ks)
    rule_curve = precision_curve(rule, y, ks)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(ks, forest_curve, color=FOREST, linewidth=2, label="Forest")
    ax.plot(ks, rule_curve, color=RULE, linewidth=2, label="Fair rule")
    ax.axhline(float(y.mean()), color=BASE, linestyle="--", linewidth=1.2, label=f"Fold base rate ({y.mean():.3f})")
    ax.axvline(50, color="black", linestyle=":", linewidth=1)
    ax.scatter([50], [precision_at_k(forest, y, 50)], color=FOREST, zorder=5)
    ax.scatter([50], [precision_at_k(rule, y, 50)], color=RULE, zorder=5)
    ax.set_xlabel("K (top of ranked list)")
    ax.set_ylabel("Precision@K")
    ax.set_ylim(0, 1.0)
    ax.set_title("Precision@K on client-holdout fold 1 (reported metric family)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "precision_recall.png")


def plot_error(forest, rule, y) -> None:
    forest_hit = int(round(precision_at_k(forest, y, 50) * 50))
    rule_hit = int(round(precision_at_k(rule, y, 50) * 50))
    labels = ["Fair rule", "Forest"]
    hits = [rule_hit, forest_hit]
    misses = [50 - rule_hit, 50 - forest_hit]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, hits, 0.55, label="Declining in top 50", color=MEAN, edgecolor="black", linewidth=0.6)
    ax.bar(x, misses, 0.55, bottom=hits, label="Not declining in top 50", color="#e8c4c4", edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pages in top 50")
    ax.set_ylim(0, 60)
    ax.set_title("Top-50 composition on fold 1 (Precision@50)")
    for i, (h, m) in enumerate(zip(hits, misses)):
        ax.text(i, h / 2, str(h), ha="center", va="center", fontsize=11)
        ax.text(i, h + m / 2, str(m), ha="center", va="center", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "error_analysis.png")


def plot_distribution(df) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    declining = df["is_declining"] == 1
    panels = [
        (axes[0, 0], "ctr", "CTR (stored x100)", (0, 8)),
        (axes[0, 1], "avg_position", "Average position", (0, 50)),
        (axes[1, 0], "impressions_90d", "Log(impressions + 1)", None),
        (axes[1, 1], "content_age_days", "Content age (days)", (0, 4000)),
    ]
    for ax, col, xlabel, xlim in panels:
        a = df.loc[~declining, col].fillna(0)
        b = df.loc[declining, col].fillna(0)
        if col == "impressions_90d":
            a = np.log1p(a.clip(lower=0))
            b = np.log1p(b.clip(lower=0))
        ax.hist(a, bins=40, alpha=0.55, label="Not declining", color=FOREST, range=xlim)
        ax.hist(b, bins=40, alpha=0.55, label="Declining", color=RULE, range=xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Warehouse feature distributions by Apr-vs-Mar label", y=0.98)
    fig.tight_layout()
    _save(fig, "distribution.png")


def main() -> None:
    m = _metrics()
    print("loading warehouse frame…")
    df = load_page_frame()
    print("training fold-1 forest for Precision@K…")
    forest, rule, y = _fold1(df)
    plot_performance(m)
    plot_comparison(m)
    plot_importances(m)
    plot_precision_at_k(forest, rule, y)
    plot_error(forest, rule, y)
    plot_distribution(df)
    print("done")


if __name__ == "__main__":
    repo_root()
    main()
