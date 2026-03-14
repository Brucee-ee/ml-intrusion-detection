"""
Phishing URL Detector — Training Pipeline
Luke - CS @ Swansea Uni, second year

Uses the UCI Phishing Dataset (id=967) which has pre-extracted features
covering URL structure, page content, and domain info.
Switched to this from the Kaggle dataset which had completely backwards labels —
"good" URLs were actually the phishing ones. Classic.

Setup:
    pip install scikit-learn pandas numpy matplotlib seaborn joblib ucimlrepo
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "./phishing_model.pkl"
SEED       = 42

# ── 1. Load Dataset ───────────────────────────────────────────────────────────
# UCI dataset 967 — 50 pre-extracted features per URL
# covers URL structure, HTML content analysis, and domain reputation
# label: 1 = legitimate, 0 = phishing
# way cleaner than rolling our own features from raw URL strings
print("[*] Loading UCI Phishing dataset...")
phishing = fetch_ucirepo(id=967)
df = phishing.data.original

print(f"    Shape: {df.shape}")
print(f"    Label distribution:\n{df['label'].value_counts()}")

# ── 2. Prepare Features ───────────────────────────────────────────────────────
# drop columns that aren't useful for the model
# FILENAME and Title are strings, URL we don't need since features are already extracted
# TLD is also a string (e.g. ".com", ".xyz") — would need encoding to use
drop_cols = ["FILENAME", "URL", "Title", "TLD"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# fill missing values with 0 — some pages couldn't be fetched during dataset creation
df = df.fillna(0)

X = df.drop(columns=["label"])
X = X.select_dtypes(include=[np.number])

# only keep URL-based features — page content features aren't available at inference time
# training on them causes the demo to flag everything as phishing since they're all zeroed out
url_features = [
    "URLLength", "DomainLength", "IsDomainIP", "NoOfSubDomain",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "URLCharProb", "TLDLength"
]
X = X[[c for c in url_features if c in X.columns]]

y = df["label"].values

print(f"\n    Feature matrix: {X.shape}")
print(f"    Legitimate: {(y==1).sum()} | Phishing: {(y==0).sum()}")

# ── 3. Train/Test Split ───────────────────────────────────────────────────────
# stratify=y keeps the class ratio the same in both train and test splits
# important when classes are imbalanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n    Train: {X_train.shape} | Test: {X_test.shape}")

# ── 4. Train Model ────────────────────────────────────────────────────────────
# Random Forest — solid choice for tabular data with mixed feature types
# each tree sees a random subset of features which reduces overfitting
# could swap for LightGBM like the malware project for a small accuracy boost
print("\n[*] Training Random Forest classifier...")

model = RandomForestClassifier(
    n_estimators=200,      # 200 trees — more = marginally better but slower
    max_depth=20,          # stop trees getting too deep and memorising training data
    min_samples_leaf=5,    # minimum 5 samples per leaf — another overfitting guard
    n_jobs=-1,             # use all CPU cores, speeds up training a lot
    random_state=SEED,
    verbose=1,
)
model.fit(X_train, y_train)

# save both the model and feature names so app.py knows what columns to send
joblib.dump({"model": model, "features": X.columns.tolist()}, MODEL_PATH)
print(f"    Model saved → {MODEL_PATH}")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("\n[*] Evaluating...")

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability of being legitimate (label=1)

roc_auc = roc_auc_score(y_test, y_proba)
print(f"\n    ROC-AUC: {roc_auc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["Phishing", "Legitimate"]))

# ── 6. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Phishing URL Detector — Random Forest (UCI Dataset)", fontsize=14, fontweight="bold")

# confusion matrix — ideally high numbers on the diagonal, low everywhere else
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Phishing", "Legitimate"],
            yticklabels=["Phishing", "Legitimate"], ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("True Label")
axes[0].set_xlabel("Predicted Label")

# ROC curve — AUC close to 1.0 is good, 0.5 is random guessing
RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1], name=f"RF (AUC={roc_auc:.3f})")
axes[1].set_title("ROC Curve")

PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[2], name="RF")
axes[2].set_title("Precision-Recall Curve")

plt.tight_layout()
plt.savefig("phishing_results.png", dpi=150, bbox_inches="tight")
print("\n[*] Plots saved → phishing_results.png")

# ── 7. Feature Importance ─────────────────────────────────────────────────────
# with 50 features this is genuinely interesting to look at
# expect URLSimilarityIndex, TLDLegitimateProb, DomainTitleMatchScore to rank high
# these are features that require actually visiting the page — URL-only features tend to rank lower
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().tail(20).plot(kind="barh", figsize=(10, 7), color="tomato")
plt.title("Top 20 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("phishing_feature_importance.png", dpi=150, bbox_inches="tight")
print("[*] Feature importance saved → phishing_feature_importance.png")
