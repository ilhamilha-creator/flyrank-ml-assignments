# Content Refresh Prioritization: An ML Journey
## How I Built a Decision-Support System for Content Teams

---

## The Problem

**The Reality:** Content teams managing large portfolios face a common dilemma:

> "We have thousands of pages. Which ones actually need refresh? And which ones should we leave alone?"

**The Pain Points:**
- Manual review is time-consuming (4+ hours per page)
- Random selection wastes effort on stable pages
- Editorial time is expensive and limited
- Missing declining pages means lost traffic

**The Numbers:**
- Warehouse page-level frame (not the 30k starter CSV)
- Declining rate is the Apr-vs-Mar label
- Random picking equals that base rate
- **The fair rule beats random. The forest does not, on fold 1.**

---

## The Approach

**My Lane:** Refresh / Content Opportunity Scoring

**The Decision I Support:** Which pages should content editors prioritize for refresh?

**The Unit of Analysis:** One unique content page

**The Output:** A ranked queue with clear reason codes explaining WHY each page is flagged

**The Method:** Machine learning (Random Forest) trained on historical search performance signals

---

## The Data

**Source:** FlyRank Hugging Face warehouse (`hf://datasets/FlyRank/internship-warehouse`)

**Size:** 79,576 content pages, 26 clients

**Time Window:** Features Jan–Feb 2026. Label: Apr vs Mar 2026.

**Key Signals:**
- `content_age_days`: How old is the content?
- `days_since_last_update`: How long since last refresh?
- `impressions_90d`: Search visibility
- `ctr`: Click-through rate (engagement quality)
- `avg_position`: Search ranking position

**What I Excluded:**
- `trend_pct`: Label-derived, creates leakage
- Product decision flags: Would learn existing rules
- Client identifiers: Used only for validation, not features

---

## The Baseline

**Before ML, I built a transparent rule:**

```
baseline_score = 0.4 * stale_visible * impressions
               + 0.3 * position_decay_risk * impressions
               + 0.3 * low_engagement * impressions
```

**The Logic:**
- Stale visible pages: Old content that still gets traffic
- Position decay risk: Older pages sitting past position 10
- Low engagement: High impressions with very low CTR

**Baseline Performance:** Precision@50 = 0.640 (32/50 correct) on the same client-holdout test set as the model.

**This is the bar ML needs to beat.**

---

## The Model

**Method:** Random Forest Classifier (100 trees, max_depth=8)

**Why Random Forest?**
- Captures non-linear relationships between signals
- Robust to different feature scales
- Provides interpretable feature importance
- Proven effective in the starter pipeline

**The Model vs Baseline:**

| Method | Precision@50 | note |
|--------|---------------|------|
| Fair hand-written rule | 0.640 | fold 1 |
| Random Forest (5-fold mean) | 0.544 | ± 0.199 |
| Random Forest (fold 1) | 0.280 | one client |
| Test-fold base rate | 0.439 | fold 1 |

**The Result:** On this warehouse future label, the fair rule beats the forest on fold 1. Five-fold mean is near the 0.557 declining rate.

---

## The Validation

**Honest Split:** Client-holdout validation (GroupKFold)

**Why:** Pages from the same client never appear in both training and test sets. This tests generalization to unseen clients, not memorization.

**The Check:**
- Fold 1 Precision@50: fair rule 0.640, forest 0.280 (one client, 20,793 pages)
- Five-fold mean forest 0.544
- Train: 58,783 / Test: 20,793 (GroupKFold, first fold)
- Same five features, `random_state=42`

**The Takeaway:** The number we report is from unseen clients, not a random page split.

---

## The Errors

**Where the model gets it wrong:**

**False Positives in the top 50 (fold 1 forest):** 36/50 were not declining (Precision@50 = 0.280)
- Fold 1 is one client
- **Cost:** Wasted editorial time if you ship the forest shortlist instead of the fair rule

**The Top 5 Features (warehouse fold 1):**
1. `days_since_last_update` (0.422)
2. `content_age_days` (0.246)
3. `ctr` (0.147)
4. `impressions_90d` (0.103)
5. `avg_position` (0.082)

---

## The Action Playbook

**The Output:** Four labels are defined. On this warehouse run only two appeared: CONTENT_REFRESH_PRIORITY (18,699 pages, 23.5%) and MONITOR_STABLE (60,877 pages, 76.5%). STALE_VISIBLE and LOW_VISIBILITY were empty.

**1. CONTENT_REFRESH_PRIORITY** (Top Priority)
- High model score and good visibility
- Review these first

**2. CONTENT_REFRESH_MODERATE** (Medium Priority)
- Old content that still gets traffic
- Refresh when time allows
- Empty on this warehouse run

**3. MONITOR_STABLE** (No Action)
- Model score is not high, and the page is not the stale-visible case
- Leave these unless a human has another reason

**4. REVIEW_THRESHOLD** (Low Priority)
- Very low visibility
- Technical SEO before a content rewrite
- Empty on this warehouse run

---

## The ROI

**Time Savings Calculation:**
- Editorial time per page: 4 hours
- Top 50 from the fair rule on fold 1: 32 declining pages (0.640)
- Top 50 at the fold base rate: about 22 (0.439)
- The forest on that fold is 0.280 — do not use it as the ROI story
- That is time allocation, not a claim that a rewrite will raise rank

**Traffic Impact:**
- Prioritizing visible declining pages means refresh efforts focus where they can recover the most organic traffic

**Team Productivity:**
- Less time debating which pages to refresh
- More time executing refreshes
- Clear reason codes explain WHY each page is flagged

---

## The Limitations

**What this work cannot claim:**
- No causal claims about ranking improvements
- No prediction of Google algorithm updates
- No guarantee that refresh will improve performance
- No knowledge of seasonal trends or competitor actions

**The Label:**
- Positive class = Apr impressions < 80% of Mar
- Features are Jan–Feb 2026, so this is a next-month drop, not a same-window proxy
- Ranking a drop is not a claim that a refresh will recover traffic

**The Single-Channel Data:**
- Only considers organic search signals
- Doesn't capture other traffic sources
- May miss patterns visible in other channels

---

## How to Use This Tomorrow

**For Content Teams:**
1. Access the ranked queue CSV
2. Review top 10-20 priority pages
3. Verify against the human review checklist:
   - Seasonal patterns?
   - Brand criticality?
   - Technical issues?
   - Content relevance?
4. Execute refresh on the most promising candidates
5. Monitor performance over 30-90 days

**The Guardrails:**
- Never automate decisions on core brand pages
- Always review legal/compliance content manually
- Check for seasonal patterns before acting
- Verify technical SEO isn't causing the decline

---

## The Monitoring

**When to Retrain:**
- Feature distribution shifts ±15%
- Declining rate changes ±10 percentage points
- Precision@50 drops below 80% of baseline
- Human feedback shows >25% false positives

**The Process:**
1. Collect new 90-day data
2. Re-run the pipeline
3. Re-validate with client-holdout
4. Compare against baseline
5. Update thresholds if needed

---

## The Takeaway

**What I Built:**
A decision-support system that helps content teams prioritize refresh efforts by identifying pages showing observed decline patterns.

**What It Does:**
- Fair rule 0.640 Precision@50 vs forest 0.280 on fold 1 (five-fold mean 0.544)
- Gives reason codes so editors can see why a page was queued
- The rule is the stronger shortlist on this warehouse future label

**What It Doesn't Do:**
- Guarantee ranking improvements
- Predict algorithm updates
- Replace human judgment
- Work without monitoring and retraining

**The Value:**
This isn't automation—it's prioritization. Content teams still make the final decisions, but now they make them with data-driven insights instead of guessing.

---

## Acknowledgments

Built on the FlyRank ML Internship dataset from [FlyRank](https://flyrank.ai). The starter dataset and lane guide provided the foundation for this work.

**Repository:** https://github.com/ilhamilha-creator/flyrank-ml-assignments

**Paper:** https://ilhamilha-creator.github.io/flyrank-ml-assignments/paper/
