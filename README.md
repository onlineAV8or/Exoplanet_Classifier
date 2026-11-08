# Kepler Exoplanet Classification Using Machine Learning

## A. Problem Statement

This project compares five supervised classification algorithms on labelled observations from a selected Kepler Objects of Interest (KOI) dataset. The objective is to classify astronomical observations using their measured input features and evaluate model performance on unseen, labelled test data.

Models: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine.

## B. Dataset Description

- **Dataset/source:** NASA Exoplanet Archive — Kepler Objects of Interest (KOI) *cumulative* table. Downloaded via the archive API: `https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv` (publicly available; also mirrored on Kaggle as "Kepler Exoplanet Search Results").
- **Number of instances:** 9,564 KOIs (full table). After a stratified 80/20 split, the held-out `test_data.csv` contains **1,913** rows.
- **Number of input features:** **34** numeric features (see list below).
- **Target column:** `koi_disposition`
- **Classes (multi-class, 3):** `CONFIRMED` (2,748), `CANDIDATE` (1,977), `FALSE POSITIVE` (4,839).
- **Input features (34):** transit and stellar measurements plus their reported uncertainties — `koi_period`, `koi_time0bk`, `koi_impact`, `koi_duration`, `koi_depth`, `koi_prad`, `koi_teq`, `koi_insol`, `koi_model_snr`, `koi_tce_plnt_num`, `koi_steff`, `koi_slogg`, `koi_srad`, `koi_kepmag`, and the `_err1`/`_err2` error columns for each (error columns that were entirely empty, e.g. `koi_teq_err*`, were dropped automatically).
- **Preprocessing:** numeric-only feature matrix; median imputation of missing values; `StandardScaler` for Logistic Regression, KNN, Naive Bayes and SVM (tree uses imputation only). All preprocessing is fitted on the training split only and saved inside each pipeline, so the Streamlit app never refits anything.
- **Leakage exclusions (dropped before training):**
  - `kepler_name` — assigned only to CONFIRMED planets, so it directly reveals the label.
  - `koi_pdisposition` and `koi_score` — the Kepler pipeline's own disposition and confidence toward the label.
  - `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec` — vetting flags that essentially define the FALSE POSITIVE class.
  - Identifier / non-feature columns — `kepid`, `kepoi_name`, `koi_tce_delivname`, `ra_str`, `dec_str`.

## C. GitHub Repository Link

https://github.com/onlineAV8or/Exoplanet_Classifier

The repository contains the Streamlit app, requirements, labelled test CSV, saved model pipelines, training source code and notebooks.

## D. Models Used and Performance Evaluation

All five saved pipelines are evaluated on the same uploaded labelled test CSV. The Streamlit app does not retrain models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7209 | 0.8874 | 0.7604 | 0.7209 | 0.7287 | 0.5830 |
| Decision Tree | 0.7067 | 0.7689 | 0.7109 | 0.7067 | 0.7086 | 0.5264 |
| KNN | 0.7182 | 0.8636 | 0.7136 | 0.7182 | 0.7139 | 0.5481 |
| Naive Bayes | 0.4945 | 0.7914 | 0.6655 | 0.4945 | 0.4701 | 0.3250 |
| Support Vector Machine | **0.7418** | **0.9026** | **0.7769** | **0.7418** | **0.7509** | **0.6094** |

*Metrics measured on the held-out `test_data.csv` (1,913 rows). AUC uses weighted one-vs-rest; precision, recall and F1 use weighted averaging. These are the exact numbers the Streamlit app reproduces when `test_data.csv` is uploaded.*

For multiclass classification, AUC uses weighted one-vs-rest; precision, recall and F1 use weighted averaging.

### Model observations

| Model | Observation |
|---|---|
| Logistic Regression | Strong, stable baseline (F1 0.729, AUC 0.887 — second best). The linear boundary separates FALSE POSITIVE cleanly but struggles to tell CONFIRMED from CANDIDATE, since those two classes overlap heavily in the measured features. |
| Decision Tree | Reasonable accuracy (0.707) but the **lowest AUC (0.769)**: a single tree produces coarse, step-like probability estimates, so its ranking/confidence is weak and it tends to overfit the training split despite `class_weight="balanced"`. |
| KNN | Competitive after scaling (Acc 0.718, AUC 0.864). Distance-based voting works once features are standardised, but performance is sensitive to local density and the choice of *k* (=5), leaving it mid-pack. |
| Naive Bayes | **Weakest model (Acc 0.495, F1 0.470).** The Gaussian feature-independence assumption is badly violated — the transit measurements and their paired error columns are strongly correlated and skewed — so it over-predicts minority classes. Its AUC (0.791) shows it still captures *some* ranking signal even though hard predictions are poor. |
| Support Vector Machine | **Best on every metric** (Acc 0.742, AUC 0.903, F1 0.751, MCC 0.609). The RBF kernel captures non-linear boundaries between the three dispositions, and balanced class weights handle the class imbalance well. |

**Overall winner:** **Support Vector Machine.** It ranks first across all six metrics — accuracy, AUC, precision, recall, F1 and MCC — making it the most reliable classifier on this dataset. Logistic Regression is a close, much cheaper runner-up (notably faster to train and score), while Gaussian Naive Bayes is clearly unsuited to these correlated, non-Gaussian features.

## Running locally

Tested with **Python 3.11** and the pinned versions in `requirements.txt`.

1. Install dependencies: `pip install -r requirements.txt`
2. *(Only to regenerate models)* Download the dataset to `data/kepler.csv`:
   `https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv`
   The dataset and `train_models.py` config (`TARGET_COLUMN = "koi_disposition"`, leakage `DROP_COLUMNS`) are already set up.
3. Run `python train_models.py` to (re)create the saved pipelines, `model/metadata.json` and `test_data.csv`. *(Already generated and committed — this step is optional.)*
4. Run `streamlit run app.py`.
5. Upload `test_data.csv` in the app to evaluate all five saved models.

> **Deployment note:** `scikit-learn`, `numpy` and `joblib` are pinned to the exact versions used to train the saved `.joblib` pipelines, so the deployed app loads them without version-mismatch errors.

## Using the interactive app

The app presents the whole evaluation as a guided "mission control" flow. No model is retrained in the browser — it only loads the saved pipelines and scores them on the test data you provide.

**Quick start:** in the **Telemetry Uplink** panel, either upload a labelled CSV or click **Use bundled sample** to load `test_data.csv` instantly, then press the glowing **▼ Run Diagnostics ▼** button to jump to the results.

**Left navigation panel.** A persistent sidebar links to every section. The three results sections (Fleet Diagnostics, Deep Scan, Explorer Relevance) stay **locked 🔒 until test data is loaded**; clicking a locked item takes you to the uplink panel instead. Once data is loaded they unlock and scroll to their section.

**Sections, top to bottom:**

1. **Telemetry Uplink** — choose a data source: *Upload my CSV* (your own labelled test set) or *Use bundled sample* (`test_data.csv`). A download button provides the sample so you can see the expected column format. The app checks that all required feature columns and the target column are present.
2. **Mission Objective** — states the aim of the test and how it works (five pre-trained models scored on six metrics on the same held-out set).
3. **KOI Field Guide** — a short primer on Kepler Objects of Interest, the transit method, and the three disposition classes (CONFIRMED / CANDIDATE / FALSE POSITIVE).
4. **Dataset Dossier** — a data dictionary: every column group, what it tells us, whether it is used as a feature, and why (including the leakage exclusions).
5. **Fleet Diagnostics — Performance Matrix** — the comparison table of all five models across Accuracy, AUC, Precision, Recall, F1 and MCC, a grouped bar chart, and an automatic "fleet winner" banner.
6. **Deep Scan** — pick any model from the highlighted selector to see its six metric tiles, **confusion matrix**, **classification report**, a **per-class detection** chart, and a **Mission Debrief** that reads the confusion matrix in plain language (e.g. share of real planets recovered vs. false positives rejected).
7. **Why This Matters to Space Explorers** — how such a classifier is used in practice to triage telescope follow-up, with an honest note on its limits.
8. **Signal Log** — actual-vs-predicted rows for the selected model, with a ✔/✘ match flag and a CSV download of the predictions.

All metrics shown in the app on `test_data.csv` match the comparison table in section **D** above.

## Deployment

Push the required files and generated `model/` artifacts to GitHub. Deploy `app.py` using Streamlit Community Cloud.

## Important

The uploaded CSV must contain the exact feature columns and target column used during training. Do not use target-derived features or identifiers that leak the answer. Only upload a held-out test set for evaluation.
