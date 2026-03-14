"""
Phishing URL Detector — Streamlit Demo
Luke - CS @ Swansea Uni, second year

Run with: streamlit run app.py
Make sure phishing_model.pkl is in the same folder (run train.py first)

Note: the full model uses 50 features including page content (HTML, JS, CSS counts etc.)
In production you'd run a headless browser to fetch those features first.
The live demo here uses URL-only features — accuracy is lower but works for obvious cases.
For batch analysis with full features, use the trained model directly.
"""

import re
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import urlparse

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🎣",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    bundle = joblib.load("./phishing_model.pkl")
    return bundle["model"], bundle["features"]

model, feature_names = load_model()

# ── Feature extraction ────────────────────────────────────────────────────────
def get_entropy(s):
    if not s:
        return 0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    probs = [f / len(s) for f in freq.values()]
    return -sum(p * math.log2(p) for p in probs)

def extract_features(url):
    try:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
        domain = parsed.netloc or ""
        path   = parsed.path or ""
        query  = parsed.query or ""
    except Exception:
        parsed = urlparse("")
        domain, path, query = "", "", ""

    known = {
        "URLLength":               len(url),
        "DomainLength":            len(domain),
        "IsDomainIP":              int(bool(re.match(r"(\d{1,3}\.){3}\d{1,3}", domain))),
        "NoOfSubDomain":           domain.count("."),
        "NoOfLettersInURL":        sum(c.isalpha() for c in url),
        "LetterRatioInURL":        sum(c.isalpha() for c in url) / max(len(url), 1),
        "NoOfDegitsInURL":         sum(c.isdigit() for c in url),
        "DegitRatioInURL":         sum(c.isdigit() for c in url) / max(len(url), 1),
        "NoOfEqualsInURL":         url.count("="),
        "NoOfQMarkInURL":          url.count("?"),
        "NoOfAmpersandInURL":      url.count("&"),
        "NoOfOtherSpecialCharsInURL": sum(not c.isalnum() and c not in "/:.-_?=&" for c in url),
        "SpacialCharRatioInURL":   sum(not c.isalnum() for c in url) / max(len(url), 1),
        "IsHTTPS":                 int(parsed.scheme == "https"),
        "HasObfuscation":          int("%" in url or "@" in url),
        "NoOfObfuscatedChar":      url.count("%") + url.count("@"),
        "ObfuscationRatio":        (url.count("%") + url.count("@")) / max(len(url), 1),
        "URLCharProb":             get_entropy(url),
        "TLDLength":               len(domain.split(".")[-1]) if "." in domain else 0,
    }
    row = {f: known.get(f, 0) for f in feature_names}
    return row

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎣 Phishing URL Detector")

st.info(
    "**Model:** Random Forest trained on UCI Phishing Dataset (235k URLs, 0.9986 ROC-AUC)  \n"
    "**Live demo limitation:** Full model uses 50 features including page content analysis "
    "(HTML structure, JS/CSS counts, form fields etc.). Live demo uses URL-only features — "
    "in production a headless browser would fetch page features before classification.",
    icon="ℹ️"
)

url_input = st.text_input("URL", placeholder="https://example.com/login")

if url_input:
    features = extract_features(url_input)
    X = pd.DataFrame([features])

    proba    = model.predict_proba(X)[0][1]
    is_legit = proba >= 0.85

    if is_legit:
        st.success(f"✅ **Likely Legitimate** — {proba*100:.1f}% confidence")
    else:
        st.error(f"⚠️ **Likely Phishing** — {(1-proba)*100:.1f}% confidence")

    st.progress(float(1 - proba), text=f"Phishing probability: {(1-proba):.2%}")

    with st.expander("Feature breakdown"):
        df_features = pd.DataFrame(features.items(), columns=["Feature", "Value"])
        st.dataframe(df_features, width="stretch")

# ── Bulk check ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Bulk Check")
st.markdown("Paste multiple URLs (one per line):")

bulk_input = st.text_area("URLs", height=150, placeholder="https://example.com\nhttps://suspicious-site.com/login?id=123")

if st.button("Scan All") and bulk_input:
    urls = [u.strip() for u in bulk_input.strip().split("\n") if u.strip()]
    results = []
    for url in urls:
        feat = extract_features(url)
        prob = model.predict_proba(pd.DataFrame([feat]))[0][1]
        results.append({
            "URL":        url,
            "Verdict":    "✅ Legitimate" if prob >= 0.85 else "🚨 Phishing",
            "Phishing %": f"{(1-prob)*100:.1f}%"
        })
    st.dataframe(pd.DataFrame(results), width="stretch")

st.divider()
st.caption("Random Forest · UCI Phishing Dataset · 0.9986 ROC-AUC | COMP-SCI @ Swansea Uni")
