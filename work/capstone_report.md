# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** ilhamilha-creator
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/ilhamilha-creator/flyrank-ml-assignments
- **Date:** 2026

> Copy this file to `work/capstone_report.md` and fill it in as you build. Sections 1–8
> mirror the Pass / Needs-Work rubric axes, so nothing here is optional. Sections 0 and 9
> are **paper sections**: your deployed research paper must carry both, and they're here so
> you never rebuild them from memory at ship time.

## 0. Abstract

Five sentences, written last, placed first: question → data → method → headline result →
what the output is for. This is the top of your deployed paper.

Content teams managing large portfolios struggle to identify which declining pages actually need refresh before wasting editorial time on stable content. This work uses the FlyRank Hugging Face warehouse: 79,576 pages, 26 clients. Features are January–February 2026. The label is a next-month drop (April impressions < 80% of March). A fair hand-written rule reaches Precision@50 of 0.640 on client-holdout fold 1. A Random Forest on the same five features reaches 0.280 on that fold (five-fold mean 0.544). The output is a ranked action queue with reason codes. This is decision-support for prioritization, not a predictor of ranking gains.

## 1. Problem framing

What decision does this support? Name the unit of analysis (page, client, day…), the output
(score, rank, cluster, report), the action a human takes from it, and the cost of a wrong
call. Why does data/ML help here at all?

**Decision supported:** Which content pages should content editors prioritize for refresh to recover declining search performance?

**Unit of analysis:** One unique content page (identified by `content_id`)

**Output:** Ranked priority queue with action labels and reason codes. Four labels are defined; on this warehouse run only CONTENT_REFRESH_PRIORITY (18,699) and MONITOR_STABLE (60,877) appeared.

**Human action:** Content editors review flagged pages in priority order, assess whether they actually need refresh, and execute updates on the most promising candidates

**Cost of wrong call:**
- **False positive:** Editorial time wasted on stable pages that don't need refresh (~4 hours per page)
- **False negative:** Missed opportunity to recover declining pages that could have been improved

**Why data/ML helps:** Random picking matches the declining rate (0.557). On this warehouse a fair hand-written rule is the stronger shortlist (Precision@50 0.640 on fold 1). The Random Forest did not beat that rule (0.280). The queue is still ranked decision-support with reason codes, not a claim that ML beats a transparent rule.

## 2. Data safety

Which data you used and which columns you deliberately excluded (and why). Leakage risks you
considered — especially label-derived fields (`trend_direction`, `trend_pct`) and pseudonymous
IDs (grouping only, never features). Confirm nothing client-identifying appears anywhere in
`work/`.

**Data source:** FlyRank Hugging Face warehouse (`hf://datasets/FlyRank/internship-warehouse`) — 79,576 pages, 26 clients. Features Jan–Feb 2026. Label: Apr impressions < 80% of Mar.

**Time window:** Features Jan–Feb 2026. Label: Apr impressions < 80% of Mar. Decision date 2026-03-01.

**Grain:** One unique content page (`content_id`)

**Scope:** 26 pseudonymized clients, warehouse page frame (79,576 pages).

**Columns deliberately excluded:**
- `trend_pct`: Label-derived feature computed from trend_direction, excluded to prevent leakage
- Product decision flags (if present): health_score, priority_score, action_type, refresh_tier excluded to avoid learning existing system rules
- Client identifiers: `client_id` used only for grouped validation splits, never as features
- Content URLs and page titles: Excluded to maintain privacy and avoid overfitting to specific content

**Leakage risks considered:**
- Label-derived features: `trend_pct` deliberately tested and confirmed leaky (inflates performance by ~15%), excluded from final model
- Future window overlap: Features are Jan–Feb 2026. Label is Apr vs Mar. Features sit before the label window.
- Decision-derived features: No product flags or existing system scores used as inputs

**Privacy confirmation:** No client names, URLs, raw queries, or private data appear in any work notebooks or outputs. All work uses anonymized identifiers only.

## 3. Baseline

The transparent rule or score you built first. Why it's a fair comparison, and its numbers on
the same data and metric as your model.

**Transparent baseline rule:** A hand-written rule combining three signals (NO label leakage):
- Stale visible pages: `days_since_last_update >= 180` AND `impressions_90d >= 500`
- Position decay risk: `avg_position > 10` AND `content_age_days >= 180`
- Low engagement high visibility: `ctr < 0.05` AND `impressions_90d >= 1000`

**Score formula:**
```
baseline_score = 0.4 * stale_visible * impressions_90d
               + 0.3 * position_decay_risk * impressions_90d
               + 0.3 * low_engagement * impressions_90d
```

**Reason codes:** STALE_VISIBLE, POSITION_DECAY_RISK, LOW_ENGAGEMENT, MONITOR

**Why fair comparison:** The baseline uses ONLY knowable-at-decision-time signals (age, freshness, impressions, position, CTR) and does NOT use any label-derived features (trend_direction, trend_pct) or product flags. It represents the "current practice" that ML should improve upon.

**Baseline performance:** Precision@50 = 0.640 (32/50 pages correctly identified as declining) on the same client-holdout test set used for model evaluation. This is the result from executing w04_baseline_score.ipynb with fair baseline rules.

## 4. Model / analysis

Your method and why it fits the lane. The exact feature list (and what you left out on
purpose). The target or proxy definition, in one sentence.

**Method:** Random Forest Classifier (100 trees, max_depth=8, random_state=42)

**Why it fits the lane:**
- Non-linear relationships: Content performance involves complex interactions between age, freshness, position, and engagement
- Robust to scaling: Features have different scales (impressions in thousands, CTR as decimal); tree models don't require normalization
- Feature importance: Built-in interpretability helps explain which signals matter most
- Readable enough: feature importance shows which signals the model leaned on, which a pure black-box would not

**Feature list (5 core features):**
1. `content_age_days`: How old the content is ( freshness signal)
2. `days_since_last_update`: How long since last modification (stale content risk)
3. `impressions_90d`: Search visibility over 90 days (traffic potential)
4. `ctr`: Click-through rate (engagement quality)
5. `avg_position`: Average search ranking position (visibility tier)

**Excluded on purpose:**
- `trend_pct`: Label-derived, creates leakage
- Product decision flags: Would learn existing rules rather than discover patterns
- `trend_direction`: Used as label, not as feature (model should learn to predict decline from other signals)

**Target definition:** Binary classification. Positive class = Apr impressions < 80% of Mar. Features are Jan–Feb 2026, so this is a next-month drop, not a same-window proxy. The stored column is still named `trend_direction` / `is_declining`.

## 5. Evaluation

Your split (grouped by client? time-aware?) and why. Metrics, model vs baseline **on the same
**split**. What the errors look like — a short error analysis beats a big metric table.

**Split design:** Client-holdout validation using GroupKFold (5 folds). Pages from the same client are never in both training and test sets, ensuring the model generalizes to unseen clients rather than memorizing client-specific patterns.

**Why this split:** A random split would allow the model to learn client-specific patterns that won't generalize. Client-holdout tests the honest question: "does this work on a client we've never seen?"

**Metrics (model vs baseline on same split):**

| Method | Precision@50 |
|--------|---------------|
| Fair Hand-written Baseline | 0.640 |
| Random Forest (5-fold mean) | 0.544 |
| Random Forest (fold 1) | 0.280 |
| Test-fold base rate | 0.439 |

**Key finding:** On the Hugging Face warehouse the fair rule (0.640) beats the forest on fold 1 (0.280). Five-fold mean for the forest is 0.544, near the 0.557 declining rate. The CSV same-window proxy had made the forest look stronger.

**Error analysis:**
- Fold 1 is one client (20,793 pages). 0.280 means 14/50 in the top list match the Apr-vs-Mar drop.
- The fair rule on that fold is 32/50.

**Feature importance (warehouse fold 1):**
1. `days_since_last_update` (0.422)
2. `content_age_days` (0.246)
3. `ctr` (0.147)
4. `impressions_90d` (0.103)
5. `avg_position` (0.082)

## 6. Interpretation

What the model/clusters actually found. Feature importances or cluster profiles in plain
words. Surprises and negative results — a well-understood "no effect" is a valid result.

**What the model found:** On warehouse fold 1 the forest leaned hardest on `days_since_last_update`, then `content_age_days`. CTR, impressions, and position were weaker. That is the opposite of the starter-CSV story, where visibility and rank led.

**Feature interpretation in plain words:**
- **days_since_last_update**: Strongest forest weight on this fold. Some values are negative (update after the 2026-03-01 decision date), so treat this signal with care.
- **content_age_days**: Age is the second forest weight. Old is not the same as declining.
- **ctr**: Third. Engagement helped a little.
- **impressions_90d**: Fourth. Visibility still matters for who is worth reviewing.
- **avg_position**: Weakest of the five on this fold.

**Surprises:**
- The forest lost to the fair rule on fold 1 (0.280 vs 0.640)
- Staleness ranked first here, last on the starter CSV
- Fold 1 is one client, so importances can move across folds

**Negative results:**
- Adding `trend_pct` (label-derived) inflates performance and is leakage, so it stayed out
- Random Forest did not beat the fair rule on this future label. Logistic Regression scored 0.780 on the same fold, but the reported model remains the forest from Week 5.

## 7. Recommendation

The ranked actions or decisions your output supports, and how a FlyRank editor would use them
tomorrow. State your confidence and the limits explicitly.

**Ranked actions for content teams:**

The queue is ranked by the model's decline score (out-of-fold, grouped by client). Action labels do not look up `trend_direction`. Cohort sizes are computed in `w07_action_playbook.ipynb`.

1. **CONTENT_REFRESH_PRIORITY** (MODEL_RISK_HIGH_VISIBILITY): Model score in the top 30%, `impressions_90d ≥ 100`, and `avg_position` between 1 and 20. Review these first — visible pages the model flags as looking like decline. On this warehouse run: 18,699 pages.

2. **CONTENT_REFRESH_MODERATE** (STALE_VISIBLE): `days_since_last_update ≥ 180` and `impressions_90d ≥ 500`, and not already in priority. Old pages that still get traffic. On this warehouse run: 0 pages.

3. **MONITOR_STABLE** (MONITOR): Everything else with some visibility. Leave these unless a human has another reason to touch them. On this warehouse run: 60,877 pages.

4. **REVIEW_THRESHOLD** (LOW_VISIBILITY): `impressions_90d < 50`. Content refresh is the wrong first move; visibility is the issue. On this warehouse run: 0 pages.

**How a FlyRank editor would use this tomorrow:**

1. **Access the ranked queue** from `work/outputs/` after running `w07_action_playbook.ipynb`
2. **Review top 10-20 pages** in CONTENT_REFRESH_PRIORITY category
3. **Verify against human review checklist:**
   - Seasonal patterns? (is decline normal for this time of year?)
   - Brand criticality? (is this a core brand page?)
   - Technical issues? (are there crawl errors or indexing problems?)
   - Content relevance? (is the content still accurate and useful?)
4. **Execute refresh** on the most promising candidates
5. **Monitor performance** over the next 30-90 days

**ROI estimate:**
- Editorial time per page: ~4 hours
- Top 50 from the **fair rule** on fold 1: 32 declining pages (0.640)
- Top 50 at the fold base rate: about 22 (0.439)
- That is time allocation, not a claim that a rewrite will raise rank. The forest on that fold is 0.280, so do not use the forest shortlist as the ROI story.

**Confidence:** The fair rule beats random on fold 1 (0.640 vs 0.439). The forest does not (0.280). Medium confidence in the ranked queue as decision-support. Human review is required before any rewrite.

## 8. Reproducibility

The exact commands to re-run everything from a fresh clone, your random seeds, and your
environment (`pip freeze` highlights or `requirements.txt` deltas). If you claim a sealed or
holdout evaluation, two things must be committed: the cell/script that builds the sealed
frame, AND the metrics file it produced — "evaluated once, blind" should be checkable from
your repo, not taken on faith.

**To re-run from a fresh clone:**

```bash
git clone https://github.com/ilhamilha-creator/flyrank-ml-assignments
cd flyrank-ml-assignments
pip install -r requirements.txt
```

**Run notebooks in order:**
1. `work/notebooks/w01_research_question.ipynb`
2. `work/notebooks/w02_ml_task_framing.ipynb`
3. `work/notebooks/w03_data_contract.ipynb`
4. `work/notebooks/w04_baseline_score.ipynb`
5. `work/notebooks/w05_model.ipynb`
6. `work/notebooks/w06_validation_audit.ipynb`
7. `work/notebooks/w07_action_playbook.ipynb`

**Random seeds:** All models use `random_state=42` for reproducibility

**Environment:** Python 3.x with pandas, numpy, scikit-learn, jupyter

**Output files:**
- `work/outputs/canonical_metrics.json`: Committed metrics receipt (Precision@50, importances, split sizes)
- `work/outputs/baseline_action_score.csv`: Ranked baseline queue (written by `w04_baseline_score.ipynb`)
- `work/outputs/action_playbook_queue.csv`: Model playbook queue (written by `w07_action_playbook.ipynb`)
- `work/outputs/action_playbook_baseline_metrics.json`: Monitoring medians (written by `w07_action_playbook.ipynb`)
- `work/capstone_report.md`: This report
- `work/presentation.md`: Short talk outline
- `docs/paper/`: Deployed paper and charts

## 9. Acknowledgments & data credit

One short section at the bottom of the deployed paper: "Built on the FlyRank ML Internship
dataset" **linking to https://flyrank.ai**. Crediting your data source is standard research
practice — and it's on the capstone's required-section list, so a paper without it isn't done.

Built on the FlyRank ML Internship dataset from [FlyRank](https://flyrank.ai). The starter dataset and lane guide provided the foundation for this work on content refresh prioritization using machine learning.

**Repository:** https://github.com/ilhamilha-creator/flyrank-ml-assignments

**Deployed Paper:** https://ilhamilha-creator.github.io/flyrank-ml-assignments/paper/

---

**Claims checklist before submitting:** observed / measured / directional / decision-support
**Metrics vs. base rate:** report your task's base rate (majority-class %) next to any
precision@K or accuracy — a high score can just be a high base rate. AUC / lift over
baseline are the honest discrimination numbers.
language everywhere · no causal claims without an experiment or causal design · no
"predicted Google's algorithm" · no client-identifying details · numbers in this report
match a fresh re-run.
