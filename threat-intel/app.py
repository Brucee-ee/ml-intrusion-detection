"""
Threat Intel NLP — Streamlit Demo
Luke - CS @ Swansea Uni, second year

Paste a CVE description and get a predicted severity level.
Good demo because you can grab real CVE descriptions from https://nvd.nist.gov
and see if the model agrees with the official CVSS rating.

Run with: streamlit run app.py
Make sure threat_model.pkl is in the same folder (run train.py first)
"""

import joblib
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CVE Severity Classifier",
    page_icon="🔐",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    bundle = joblib.load("./threat_model.pkl")
    return bundle["model"], bundle["tfidf"], bundle["le"]

model, tfidf, le = load_model()

# colour map for severity levels
SEVERITY_COLOURS = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
}

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔐 CVE Severity Classifier")
st.markdown(
    "Paste a vulnerability description to predict its severity level. "
    "Trained on 20,000 CVEs from the NVD — no CVSS score needed at inference time."
)

# example descriptions to make it easy to demo
with st.expander("Example descriptions to try"):
    st.markdown("""
**Critical (RCE):**
> A remote code execution vulnerability exists in the Windows Print Spooler service when it improperly performs privileged file operations. An attacker who successfully exploited this vulnerability could run arbitrary code with SYSTEM privileges.

**High:**
> An SQL injection vulnerability in the login page of ExampleCMS 4.2 allows remote attackers to bypass authentication and gain administrative access via crafted input in the username field.

**Medium:**
> A cross-site scripting (XSS) vulnerability in ExampleApp allows authenticated attackers to inject arbitrary web script via the profile name parameter.

**Low:**
> A information disclosure vulnerability exists when the browser caches sensitive data. An attacker with physical access to the machine could view cached credentials.
    """)

desc_input = st.text_area("Vulnerability description", height=150,
                           placeholder="Paste a CVE description here...")

if st.button("Classify") and desc_input:
    # vectorize the input and predict
    X = tfidf.transform([desc_input])
    pred       = model.predict(X)[0]
    proba      = model.predict_proba(X)[0]
    severity   = le.inverse_transform([pred])[0]
    confidence = proba.max()

    icon = SEVERITY_COLOURS.get(severity, "⚪")

    # colour code the result box by severity
    if severity == "CRITICAL":
        st.error(f"{icon} **{severity}** — {confidence*100:.1f}% confidence")
    elif severity == "HIGH":
        st.warning(f"{icon} **{severity}** — {confidence*100:.1f}% confidence")
    elif severity == "MEDIUM":
        st.info(f"{icon} **{severity}** — {confidence*100:.1f}% confidence")
    else:
        st.success(f"{icon} **{severity}** — {confidence*100:.1f}% confidence")

    # probability breakdown across all 4 classes
    st.subheader("Confidence breakdown")
    prob_df = pd.DataFrame({
        "Severity": le.classes_,
        "Probability": proba
    }).sort_values("Probability", ascending=False)

    for _, row in prob_df.iterrows():
        icon = SEVERITY_COLOURS.get(row["Severity"], "⚪")
        st.progress(float(row["Probability"]),
                    text=f"{icon} {row['Severity']}: {row['Probability']*100:.1f}%")

st.divider()
st.caption("Logistic Regression + TF-IDF · NVD CVE Dataset · COMP-SCI @ Swansea Uni")
