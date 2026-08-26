"""Page-level frame from the FlyRank Hugging Face warehouse.

Feature window: Jan–Feb 2026 (does not overlap the label).
Label window: Apr vs Mar 2026 (down if Apr impressions < 80% of Mar).
Token: HF_TOKEN from a .env file or the environment. Never printed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

REL = "hf://datasets/FlyRank/internship-warehouse"
FEATURE_MONTHS = ("2026-01", "2026-02")
LABEL_PRIOR_MONTH = "2026-03"
LABEL_MONTH = "2026-04"
FEATURE_START = "2026-01-01"
FEATURE_END = "2026-03-01"  # exclusive
LABEL_PRIOR_START = "2026-03-01"
LABEL_PRIOR_END = "2026-04-01"
LABEL_START = "2026-04-01"
LABEL_END = "2026-05-01"
DECISION_DATE = "2026-03-01"
MIN_FEAT_IMPRESSIONS = 50
MIN_LABEL_PRIOR_IMPRESSIONS = 50
CACHE_NAME = "warehouse_page_frame.parquet"
META_NAME = "warehouse_frame_meta.json"
FEATURES = [
    "content_age_days",
    "days_since_last_update",
    "impressions_90d",
    "ctr",
    "avg_position",
]


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "work" / "scripts").exists() and (p / "AGENTS.md").exists():
            return p
    raise FileNotFoundError("Could not find repo root (needs AGENTS.md and work/scripts)")


def cache_path(root: Path | None = None) -> Path:
    root = root or repo_root()
    return root / "work" / "outputs" / CACHE_NAME


def _read_cache(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError:
        posix = path.resolve().as_posix()
        return duckdb.sql(f"SELECT * FROM read_parquet('{posix}')").df()


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        con = duckdb.connect()
        con.register("frame", df)
        posix = path.resolve().as_posix()
        con.execute(f"COPY frame TO '{posix}' (FORMAT PARQUET)")


def _parse_env_file(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip().upper().replace("-", "_")
            value = value.strip().strip('"').strip("'")
            if key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"} and value:
                return value
        elif line.startswith("hf_"):
            return line
    return None


def load_hf_token(root: Path | None = None) -> str:
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    start = (root or Path.cwd()).resolve()
    empty_env = False
    for p in [start, *start.parents]:
        env_file = p / ".env"
        if env_file.exists():
            token = _parse_env_file(env_file)
            if token:
                return token
            if env_file.stat().st_size == 0:
                empty_env = True
    if empty_env:
        raise RuntimeError(
            "Found an empty .env. Save one line HF_TOKEN=... (Hugging Face read token) and rerun."
        )
    raise RuntimeError("HF_TOKEN not found in .env or environment")


def connect(root: Path | None = None):
    token = load_hf_token(root)
    con = duckdb.connect()
    escaped = token.replace("'", "''")
    con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{escaped}')")
    return con, REL


def _month_glob(month: str) -> str:
    return f"{REL}/fact_content_daily_performance/month={month}/*.parquet"


def _daily_src(months: tuple[str, ...] | list[str]) -> str:
    paths = ", ".join(f"'{_month_glob(m)}'" for m in months)
    return f"read_parquet([{paths}])"


def _column_names(con, src: str) -> set[str]:
    return set(con.sql(f"DESCRIBE SELECT * FROM {src}").df()["column_name"].tolist())


def _pick(cols: set[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in cols:
            return name
    return None


def probe(root: Path | None = None) -> None:
    con, rel = connect(root)
    mar = f"read_parquet('{_month_glob('2026-03')}')"
    print("connected to Hugging Face warehouse")
    n, lo, hi = con.sql(
        f"SELECT COUNT(*), MIN(report_date), MAX(report_date) FROM {mar}"
    ).fetchone()
    print(f"month=2026-03 rows={n:,} dates={lo} .. {hi}")
    print("daily columns:", sorted(_column_names(con, mar)))
    for name in ("dim_content.parquet", "dim_clients.parquet"):
        src = f"read_parquet('{rel}/{name}')"
        try:
            cols = sorted(_column_names(con, src))
            n = con.sql(f"SELECT COUNT(*) FROM {src}").fetchone()[0]
            print(f"{name}: {n:,} rows, cols={cols}")
        except Exception as exc:
            print(f"{name}: unavailable ({type(exc).__name__})")


def _age_sql(cols: set[str]) -> tuple[str, str]:
    created = _pick(
        cols,
        (
            "content_created_date",
            "content_created_at",
            "created_at",
            "first_seen_at",
            "content_first_seen_at",
        ),
    )
    updated = _pick(
        cols,
        (
            "content_updated_date",
            "last_optimized_date",
            "content_updated_at",
            "updated_at",
            "last_updated_at",
            "content_last_updated_at",
            "last_modified_at",
        ),
    )
    if created:
        age = f"DATE_DIFF('day', CAST(c.{created} AS DATE), DATE '{DECISION_DATE}')"
    else:
        age = "NULL"
    if updated:
        days_since = f"DATE_DIFF('day', CAST(c.{updated} AS DATE), DATE '{DECISION_DATE}')"
    elif created:
        days_since = age
    else:
        days_since = "NULL"
    return age, days_since


def build_page_frame(root: Path | None = None, force: bool = False) -> pd.DataFrame:
    root = root or repo_root()
    out = cache_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        return _read_cache(out)

    con, rel = connect(root)
    dim_c = f"read_parquet('{rel}/dim_content.parquet')"
    dim_cl = f"read_parquet('{rel}/dim_clients.parquet')"
    months = FEATURE_MONTHS + (LABEL_PRIOR_MONTH, LABEL_MONTH)
    daily = _daily_src(months)
    content_cols = _column_names(con, dim_c)
    age_sql, days_since_sql = _age_sql(content_cols)

    sql = f"""
    WITH daily AS (
        SELECT
            client_hash_id,
            content_hash_id,
            CAST(report_date AS DATE) AS report_date,
            COALESCE(gsc_impressions, 0) AS gsc_impressions,
            COALESCE(gsc_clicks, 0) AS gsc_clicks,
            gsc_avg_position
        FROM {daily}
        WHERE gsc_impressions IS NOT NULL
    ),
    feat AS (
        SELECT
            client_hash_id,
            content_hash_id,
            SUM(gsc_impressions) AS impressions_90d,
            SUM(gsc_clicks) AS clicks_feat,
            SUM(CASE WHEN gsc_avg_position > 0 THEN gsc_avg_position * gsc_impressions ELSE 0 END)
                / NULLIF(SUM(CASE WHEN gsc_avg_position > 0 THEN gsc_impressions ELSE 0 END), 0)
                AS avg_position
        FROM daily
        WHERE report_date >= DATE '{FEATURE_START}'
          AND report_date < DATE '{FEATURE_END}'
        GROUP BY 1, 2
        HAVING SUM(gsc_impressions) >= {MIN_FEAT_IMPRESSIONS}
    ),
    lab_prior AS (
        SELECT client_hash_id, content_hash_id, SUM(gsc_impressions) AS imp_prior
        FROM daily
        WHERE report_date >= DATE '{LABEL_PRIOR_START}'
          AND report_date < DATE '{LABEL_PRIOR_END}'
        GROUP BY 1, 2
    ),
    lab AS (
        SELECT client_hash_id, content_hash_id, SUM(gsc_impressions) AS imp_label
        FROM daily
        WHERE report_date >= DATE '{LABEL_START}'
          AND report_date < DATE '{LABEL_END}'
        GROUP BY 1, 2
    )
    SELECT
        f.client_hash_id AS client_id,
        f.content_hash_id AS content_id,
        f.impressions_90d,
        100.0 * f.clicks_feat / NULLIF(f.impressions_90d, 0) AS ctr,
        COALESCE(f.avg_position, 0) AS avg_position,
        {age_sql} AS content_age_days,
        {days_since_sql} AS days_since_last_update,
        COALESCE(lab.imp_label, 0) AS imp_label,
        COALESCE(lab_prior.imp_prior, 0) AS imp_prior
    FROM feat f
    JOIN {dim_cl} cl ON cl.client_hash_id = f.client_hash_id
    LEFT JOIN {dim_c} c ON c.content_hash_id = f.content_hash_id
    LEFT JOIN lab_prior
        ON lab_prior.client_hash_id = f.client_hash_id
       AND lab_prior.content_hash_id = f.content_hash_id
    LEFT JOIN lab
        ON lab.client_hash_id = f.client_hash_id
       AND lab.content_hash_id = f.content_hash_id
    WHERE cl.gsc_data_start IS NOT NULL
      AND CAST(cl.gsc_data_start AS DATE) <= DATE '{FEATURE_START}'
      AND COALESCE(lab_prior.imp_prior, 0) >= {MIN_LABEL_PRIOR_IMPRESSIONS}
    """
    df = con.sql(sql).df()
    ratio = df["imp_label"] / df["imp_prior"].replace(0, np.nan)
    df["trend_direction"] = np.where(
        ratio < 0.8, "down", np.where(ratio > 1.2, "up", "stable")
    )
    df["is_declining"] = (df["trend_direction"] == "down").astype(int)
    df["content_age_days"] = df["content_age_days"].fillna(df["days_since_last_update"])
    df["days_since_last_update"] = df["days_since_last_update"].fillna(df["content_age_days"])
    df["ctr"] = df["ctr"].fillna(0)
    pos = df["avg_position"].fillna(0)
    df["position_tier"] = np.select(
        [pos <= 0, pos <= 3, pos <= 10, pos <= 20],
        ["unknown", "top_3", "page_1", "page_3_5"],
        default="striking",
    )
    _write_cache(df, out)
    meta = {
        "rows": int(len(df)),
        "clients": int(df["client_id"].nunique()),
        "feature_window": [FEATURE_START, FEATURE_END],
        "label_window": [LABEL_PRIOR_START, LABEL_END],
        "label_rule": "down if Apr impressions < 0.8 * Mar impressions",
        "declining_rate": float(df["is_declining"].mean()),
        "source": REL,
        "cache": str(out.relative_to(root)),
    }
    (root / "work" / "outputs" / META_NAME).write_text(json.dumps(meta, indent=2) + "\n")
    print(f"cached {len(df):,} pages / {df['client_id'].nunique()} clients -> {out}")
    print(f"declining rate: {df['is_declining'].mean():.3f}")
    return df


def load_page_frame(root: Path | None = None, force: bool = False) -> pd.DataFrame:
    df = build_page_frame(root=root, force=force)
    if "trend_pct" not in df.columns and {"imp_label", "imp_prior"} <= set(df.columns):
        ratio = df["imp_label"] / df["imp_prior"].replace(0, np.nan)
        df["trend_pct"] = 100.0 * (ratio - 1.0)
    return df


def load_notebook_frame() -> pd.DataFrame:
    """Notebooks call this. Uses the Hugging Face warehouse, cached after the first scan."""
    df = load_page_frame()
    print(
        f"Hugging Face warehouse: {len(df):,} pages, {df['client_id'].nunique()} clients"
    )
    print("Features: Jan-Feb 2026. Label: Apr impressions < 80% of Mar.")
    print(f"Declining rate: {df['is_declining'].mean():.3f}")
    return df


def precision_at_k(scores, labels, k=50) -> float:
    order = np.argsort(-np.asarray(scores))
    topk = np.asarray(labels)[order[:k]]
    return float(topk.mean())


def score_frame(df: pd.DataFrame, root: Path | None = None) -> dict:
    X = df[FEATURES].fillna(0)
    y = df["is_declining"]
    groups = df["client_id"]
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_df = X_test.copy()
    test_age = df.iloc[test_idx]["content_age_days"].fillna(0).values
    stale_visible = (
        (test_df["days_since_last_update"] >= 180) & (test_df["impressions_90d"] >= 500)
    ).astype(int)
    position_decay_risk = ((test_df["avg_position"] > 10) & (test_age >= 180)).astype(int)
    low_engagement = ((test_df["ctr"] < 0.05) & (test_df["impressions_90d"] >= 1000)).astype(int)
    baseline = (
        0.4 * stale_visible * test_df["impressions_90d"]
        + 0.3 * position_decay_risk * test_df["impressions_90d"]
        + 0.3 * low_engagement * test_df["impressions_90d"]
    )
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    model_p50 = precision_at_k(rf.predict_proba(X_test)[:, 1], y_test.values, 50)
    baseline_p50 = precision_at_k(baseline.values, y_test.values, 50)
    base_rate = float(y_test.mean())
    metrics = {
        "base_rate": base_rate,
        "fair_baseline_precision_at_50": baseline_p50,
        "random_forest_precision_at_50": model_p50,
        "model_vs_baseline_ratio": model_p50 / baseline_p50 if baseline_p50 else None,
        "model_vs_random_ratio": model_p50 / base_rate if base_rate else None,
        "feature_importance": dict(zip(FEATURES, [float(x) for x in rf.feature_importances_])),
        "model_config": {
            "n_estimators": 100,
            "max_depth": 8,
            "random_state": 42,
            "features": FEATURES,
        },
        "validation_config": {
            "method": "client_holdout",
            "split": "GroupKFold(n_splits=5)",
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
        "data_source": {
            "kind": "huggingface_warehouse",
            "release": REL,
            "feature_window": [FEATURE_START, FEATURE_END],
            "label_window": [LABEL_PRIOR_START, LABEL_END],
            "n_pages": int(len(df)),
            "n_clients": int(df["client_id"].nunique()),
        },
    }
    root = root or repo_root()
    out = root / "work" / "outputs" / "canonical_metrics.json"
    out.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {out}")
    print(
        f"P@50 model={model_p50:.3f} baseline={baseline_p50:.3f} "
        f"base_rate={base_rate:.3f} n={len(df):,}"
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    root = repo_root()
    if args and args[0] == "--probe":
        probe(root)
        return
    force = "--force" in args
    df = build_page_frame(root=root, force=force)
    if "--cv" in args:
        X = df[FEATURES].fillna(0)
        y = df["is_declining"]
        gkf = GroupKFold(n_splits=5)
        scores = []
        for i, (tr, te) in enumerate(gkf.split(X, y, df["client_id"]), 1):
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
            )
            rf.fit(X.iloc[tr], y.iloc[tr])
            p = precision_at_k(rf.predict_proba(X.iloc[te])[:, 1], y.iloc[te].values, 50)
            scores.append(p)
            print(
                f"fold {i} P@50={p:.3f} n_test={len(te):,} "
                f"clients={df.iloc[te]['client_id'].nunique()} "
                f"rate={y.iloc[te].mean():.3f}"
            )
        print(f"five-fold mean P@50={float(np.mean(scores)):.3f} ± {float(np.std(scores)):.3f}")
        return
    if "--score" in args or force:
        score_frame(df, root=root)


if __name__ == "__main__":
    main()
