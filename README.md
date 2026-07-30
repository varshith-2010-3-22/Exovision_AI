# Habitax
### Kepler tells us if it's a planet. Habitax asks if it could hold life.

**97.19% F1 · 99.73% ROC-AUC · zero label leakage** — a two-stage machine learning pipeline that separates real exoplanets from false alarms in NASA's Kepler catalog, then scores the survivors for habitability using the same Earth Similarity framework astrobiologists actually use.

---

## Inspiration

Feed our finished model a textbook Earth twin — a Sun-like star, a 365-day orbit, a springtime 290 K equilibrium temperature — and it calls it fake, at 99.5% confidence. That single result taught us more than any accuracy score, and it's the reason this project exists.

The Kepler telescope can't photograph planets — it only sees a star dim slightly, over and over. Most of those dips are lies: eclipsing binaries, background stars bleeding light into the aperture, instrument noise. Astronomers spend real telescope time chasing these down by hand. We wanted to know two things: could a model learn to make that same call, and — one layer deeper — for the ones it's confident are real, could it tell us anything about whether they might actually support life?

*(Swap in your own hook here if you have one — what got you personally into exoplanets or astrobiology — the way judges respond well to a concrete personal reason, not just the problem statement.)*

## What it does

Habitax runs on NASA's Kepler Cumulative KOI catalog — 9,564 Kepler Objects of Interest — in two stages:

**Stage 1 — Classify.** A LightGBM model decides whether a KOI is a `CONFIRMED` exoplanet or a `FALSE POSITIVE`. It trains only on the 7,586 KOIs with a resolved label; the remaining 1,978 `CANDIDATE` rows are excluded from training entirely, since their true status is still unknown and would only add label noise.

- **97.19% F1 · 99.73% ROC-AUC · 98% accuracy** on a completely untouched 1,518-row holdout set
- Every column NASA's own vetting process could have used to give away the answer — `koi_score`, `koi_pdisposition`, all four `koi_fpflag_*` flags, every ID column — was identified and stripped before training. The model has to find the signal itself.

**Stage 2 — Characterize.** For every KOI the model calls a planet, Habitax computes an Earth Similarity Index (ESI) and checks its equilibrium temperature and insolation, then sorts it into one of four tiers — from *Tier 1: Rocky Habitable World* down to *Tier 4: Non-Habitable* — plus a 0–100 Bio-Potential score.

Try the live version: **https://exovisionai.streamlit.app/**

## How we built it

- **Data:** NASA Kepler Cumulative KOI catalog — 9,564 rows × 140 raw columns.
- **Cleaning:** dropped unresolved `CANDIDATE` rows; removed ~20 leakage columns; K-Nearest-Neighbours imputation (k=5) on the seven most important stellar and transit measurements, chosen over a flat median fill because it preserves real correlations — a large planet tends to also run hot, and KNN captures that.
- **Feature engineering:** three physics-derived features added to the 98 raw ones — a transit-depth-based radius estimate, the Earth Similarity Index, and a planet/star radius ratio.
- **Modeling:** LightGBM, with `scale_pos_weight` correcting the 1.76:1 false-positive-to-planet imbalance. A 5-fold stratified CV baseline scored F1 0.9709 ± 0.0067; a 50-trial Optuna Bayesian search (TPE) then tuned 10 hyperparameters — learning rate, tree depth, leaf count, row/column subsampling, L1/L2 regularization.
- **Explainability:** SHAP `TreeExplainer` across the full feature set, for both global feature ranking and individual per-prediction breakdowns.
- **Deployment:** the trained model is wrapped in a real-time single-KOI scanner and served through a live Streamlit app.

## Challenges we ran into

**Leakage was everywhere at first.** The raw KOI table hides the answer in plain sight — `koi_score` and the false-positive flag columns are themselves outputs of NASA's manual vetting process. Leave any of them in and accuracy climbs toward 100% for the wrong reason. We built an explicit deny-list and hand-audited every remaining column before training.

**Optuna barely moved the needle — and that told us something real.** Fifty trials across ten hyperparameters improved F1 by just 0.0010 over the untuned baseline (0.9709 → 0.9719). Once leakage is gone and class imbalance is properly handled, the ceiling on this dataset is set by the physics of the signal, not the model's settings.

**The Earth-twin problem.** When we hand-typed a textbook Earth analog into the finished model — Sun-like star, 365-day orbit, 290 K — it called it a false positive at 0.5% confidence. A hot-Jupiter test case got the same treatment, at 1.7%. SHAP explained why: the single most important feature by a wide margin isn't planet radius or temperature — it's `koi_dikco_msky`, a sky-position centroid offset that flags whether the transit dip is truly centered on the target star or leaking in from a contaminating neighbor. That's exactly what a human vetter checks first. Our hand-typed row had no centroid measurement, no multi-quarter statistics, no KOI-count context — so the model, deprived of the diagnostics it relies on most, defaulted to suspicion. It wasn't wrong to be skeptical. It was reasoning exactly the way it was trained to, on an input we hadn't given it enough to reason about.

## Accomplishments that we're proud of

- A held-out F1 of 0.9719 and ROC-AUC of 0.9973, achieved with zero leakage columns in the training set — a number we can defend, not just report.
- Stage 2 goes past a benchmark score to ask a genuinely astrobiological question: not just "is this a planet," but "could it plausibly host life," using the same ESI framework astrobiologists actually use.
- A model that explains itself: SHAP shows exactly which measurements drove every prediction — and it turns out to lean on the same centroid-offset diagnostics professional Kepler vetters check by hand.
- A live, working demo, not just a notebook.

## What we learned

Clean labels and domain-appropriate features mattered far more than hyperparameter search — fifty rounds of Optuna added a tenth of a point once leakage and imbalance were actually handled. And the model's heavy reliance on instrumental diagnostics like centroid offset was the biggest surprise of the whole project: Habitax isn't really learning "planets look like this." It's learning "trustworthy signals look like this" — a subtly different, and we think more honest, thing for it to have learned.

## What's next for Habitax

- Give the real-time scanner sane fallback defaults — or an explicit "insufficient diagnostics" flag — for the centroid and multi-quarter columns it currently can't reason well without, so single-KOI predictions are as reliable as the batch holdout results.
- Extend the habitability layer with atmospheric data as JWST follow-up observations become available for more KOIs.
- Test the pipeline against TESS Objects of Interest to see whether the same features — and the same reliance on centroid diagnostics — transfer across missions.
- Publish a small public API so the habitability scoring can be reused outside the notebook.

## Built With

Python · LightGBM · Optuna · scikit-learn · SHAP · pandas · NumPy · Matplotlib/Seaborn · Google Colab · Streamlit

## Try it out

Live demo: https://habitax.streamlit.app
GitHub repo: *(add your link here)*
