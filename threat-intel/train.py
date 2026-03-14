"""
Threat Intel NLP — Severity Classifier
Luke - CS @ Swansea Uni, second year

Trains a text classifier to predict CVE severity (Critical/High/Medium/Low)
from the vulnerability description alone — no CVSS score used at inference time.

The idea: CVSS scores require manual analysis by security researchers.
If we can predict severity from the description text, we can triage new
vulnerabilities faster before they get a formal score.

Run fetch_nvd.py first to get the training data.

Setup:
    pip install scikit-learn pandas numpy matplotlib seaborn joblib tqdm
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "./nvd_cves.csv"
MODEL_PATH = "./threat_model.pkl"
SEED       = 42

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("[*] Loading CVE data...")
df = pd.read_csv(DATA_PATH)

print(f"    Shape: {df.shape}")
print(f"\n    Severity distribution:\n{df['severity'].value_counts()}")

# drop any rows with missing descriptions or severity
df = df.dropna(subset=["description", "severity"])

# only keep the 4 main severity levels — drop NONE as it's ambiguous
df = df[df["severity"].isin(["CRITICAL", "HIGH", "MEDIUM", "LOW"])]
print(f"\n    After filtering: {df.shape}")

# ── 2. Text Preprocessing ─────────────────────────────────────────────────────
# TF-IDF converts raw text into numerical feature vectors
# each dimension represents a word/ngram, weighted by how distinctive it is
# e.g. "remote code execution" will score high as it's rare but very informative
print("\n[*] Vectorizing CVE descriptions with TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=10000,    # top 10k most informative terms
    ngram_range=(1, 2),    # unigrams + bigrams — "code execution" is more informative than "code" alone
    sublinear_tf=True,     # apply log scaling to term frequencies — helps with common words
    strip_accents="unicode",
    min_df=3,              # ignore terms that appear in fewer than 3 documents
)

X = tfidf.fit_transform(df["description"])
print(f"    TF-IDF matrix: {X.shape}")

# encode labels — sklearn needs integers not strings
le = LabelEncoder()
y  = le.fit_transform(df["severity"])
print(f"    Classes: {le.classes_}")

# ── 3. Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\n    Train: {X_train.shape} | Test: {X_test.shape}")

# ── 4. Train Model ────────────────────────────────────────────────────────────
# Logistic Regression is actually a really strong baseline for text classification
# fast, interpretable, works well with TF-IDF sparse matrices
# could use a transformer (BERT) for better accuracy but LR is good enough here
# and way faster to train on a laptop
print("\n[*] Training Logistic Regression classifier...")

model = LogisticRegression(
    max_iter=1000, # give it enough iterations to converge
    C=1.0, # inverse regularization strength — smaller values = stronger regularization
    solver="lbfgs", 
    n_jobs=-1, 
    random_state=SEED,
)

model.fit(X_train, y_train)

# save model + vectorizer + label encoder — need all three at inference time
joblib.dump({"model": model, "tfidf": tfidf, "le": le}, MODEL_PATH)
print(f"    Model saved → {MODEL_PATH}")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("\n[*] Evaluating...")

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))

# macro AUC across all 4 classes
auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
print(f"    Macro ROC-AUC (OvR): {auc:.4f}")

# ── 6. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_title("CVE Severity Classifier — Confusion Matrix")
ax.set_ylabel("True Severity")
ax.set_xlabel("Predicted Severity")
plt.tight_layout()
plt.savefig("threat_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("\n[*] Confusion matrix saved → threat_confusion_matrix.png")

# ── 7. Top keywords per severity class ───────────────────────────────────────
# this is the interesting bit — what words are most associated with each severity?
# expect CRITICAL to have "remote code execution", "unauthenticated", "arbitrary"
# expect LOW to have "local", "limited", "requires interaction"
print("\n[*] Top keywords per severity class:")
feature_names = np.array(tfidf.get_feature_names_out())

for i, cls in enumerate(le.classes_):
    # get the coefficients for this class and find the top 10
    coefs = model.coef_[i]
    top10 = feature_names[np.argsort(coefs)[-10:]][::-1]
    print(f"    {cls}: {', '.join(top10)}")
