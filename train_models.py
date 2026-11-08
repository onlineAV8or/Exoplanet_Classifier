"""
Trains the five classifiers on the Kepler KOI dataset.
It creates a stratified hold-out test_data.csv and saves five fitted
preprocessing + classifier pipelines in model/.

Notes:
- DATA_PATH / TARGET_COLUMN / DROP_COLUMNS are configured below for the
  NASA Exoplanet Archive KOI cumulative table.
- DROP_COLUMNS excludes IDs and any columns that reveal or are derived
  from the target (target leakage).
- Only numeric columns are used as model input features.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ---------------- CONFIGURATION ----------------
DATA_PATH = Path("data/kepler.csv")  # NASA Exoplanet Archive KOI cumulative table
TARGET_COLUMN = "koi_disposition"
DROP_COLUMNS = [
    # Identifier columns (not predictive; some leak the label).
    "kepid",            # Kepler target ID
    "kepoi_name",       # KOI designation (string ID)
    "kepler_name",      # Only assigned to CONFIRMED planets -> direct label leakage
    "koi_tce_delivname",  # TCE delivery batch name (string ID)
    "ra_str",           # Right ascension as string
    "dec_str",          # Declination as string
    # Target-leakage columns: these encode the disposition itself.
    "koi_pdisposition",  # Pipeline's own CANDIDATE/FALSE POSITIVE call
    "koi_score",         # Disposition confidence score (0-1) toward the label
    "koi_fpflag_nt",     # False-positive vetting flag: not transit-like
    "koi_fpflag_ss",     # False-positive vetting flag: stellar eclipse
    "koi_fpflag_co",     # False-positive vetting flag: centroid offset
    "koi_fpflag_ec",     # False-positive vetting flag: ephemeris match contamination
]
TEST_SIZE = 0.20
RANDOM_STATE = 42
# -----------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found. Columns: {list(df.columns)}")

df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
df = df.dropna(subset=[TARGET_COLUMN]).copy()

y = df[TARGET_COLUMN]
X = df.drop(columns=[TARGET_COLUMN])

# Keep numeric input columns only.
X = X.select_dtypes(include=["number"]).copy()
if X.shape[1] < 12:
    raise ValueError(
        f"Only {X.shape[1]} numeric input features remain. "
        "Check DROP_COLUMNS and the dataset; the assignment requires at least 12."
    )
if y.nunique() < 2:
    raise ValueError("The target must contain at least two classes.")

# Remove columns that are entirely missing.
X = X.dropna(axis=1, how="all")
feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Each saved artifact is a complete pipeline. Scalers/imputers are fitted
# on training data only and reused unchanged by Streamlit.
scaled_preprocessor = ColumnTransformer(
    [("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), feature_columns)],
    remainder="drop"
)
tree_preprocessor = ColumnTransformer(
    [("numeric", SimpleImputer(strategy="median"), feature_columns)],
    remainder="drop"
)

models = {
    "Logistic Regression": Pipeline([
        ("preprocess", scaled_preprocessor),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ]),
    "Decision Tree": Pipeline([
        ("preprocess", tree_preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"))
    ]),
    "KNN": Pipeline([
        ("preprocess", scaled_preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ("preprocess", scaled_preprocessor),
        ("classifier", GaussianNB())
    ]),
    "Support Vector Machine": Pipeline([
        ("preprocess", scaled_preprocessor),
        ("classifier", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE))
    ]),
}

model_files = {}
summary_rows = []

for name, pipeline in models.items():
    print(f"Training {name}...")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    classes = list(pipeline.classes_)
    if len(classes) == 2:
        auc = roc_auc_score(y_test, probabilities[:, 1])
    else:
        auc = roc_auc_score(
            y_test, probabilities, labels=classes, multi_class="ovr", average="weighted"
        )

    summary_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": auc,
        "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    })

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipeline, MODEL_DIR / filename)
    model_files[name] = filename

# The app expects test_data.csv to include every feature plus the true target.
test_data = X_test.copy()
test_data[TARGET_COLUMN] = y_test
test_data.to_csv(ROOT / "test_data.csv", index=False)

metadata = {
    "target_column": TARGET_COLUMN,
    "feature_columns": feature_columns,
    "classes": [str(c) for c in sorted(y.unique(), key=lambda v: str(v))],
    "models": model_files,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE
}
with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

summary = pd.DataFrame(summary_rows)
summary.to_csv(ROOT / "training_evaluation_summary.csv", index=False)
print("\nHold-out evaluation summary:")
print(summary.to_string(index=False))
print("\nSaved models, metadata.json and test_data.csv.")
