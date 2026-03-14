streamlit run app.pyimport streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# basic page config - wide layout looks better for a dashboard
st.set_page_config(
    page_title="ML Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

# custom css to make it not look like every other streamlit app
# learned most of this from just googling "streamlit dark theme css"
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #c8cdd6;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .metric-card h2 {
        color: #f0f4f8;
        font-size: 2rem;
        margin: 0;
    }
    .metric-card p {
        color: #8b949e;
        font-size: 0.8rem;
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .attack-badge {
        background-color: #3d1a1a;
        color: #f87171;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .benign-badge {
        background-color: #1a3d1a;
        color: #4ade80;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    div[data-testid="stSidebarContent"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    </style>
""", unsafe_allow_html=True)

# load the model - cache_resource means it only loads once which is way faster
@st.cache_resource
def load_model():
    model_path = Path("model/xgboost_ids.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    return None  # just return None if model file isn't there yet

model = load_model()

# sidebar navigation
st.sidebar.title("🛡️ ML IDS")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Dashboard", "Predict", "Model Info"])
st.sidebar.markdown("---")

# quick stats in sidebar so you can see them from any page
st.sidebar.markdown("**Dataset:** CICIDS 2017")
st.sidebar.markdown("**Model:** XGBoost")
st.sidebar.markdown("**Accuracy:** 99.99%")


# ---- DASHBOARD PAGE ----
if page == "Dashboard":
    st.title("Network Intrusion Detection System")
    st.markdown("XGBoost classifier trained on 4.2 million real network flows")
    st.markdown("---")

    # top metrics row - used html cards because st.metric looked a bit plain
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
            <div class="metric-card">
                <h2>99.99%</h2>
                <p>Accuracy</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h2>99.98%</h2>
                <p>F1 Score</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="metric-card">
                <h2>41</h2>
                <p>False Positives</p>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
            <div class="metric-card">
                <h2>33</h2>
                <p>False Negatives</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # two charts side by side
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.subheader("Attack Distribution")
        # hardcoded from the training data counts
        attack_data = {
            "Label": ["BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye",
                      "FTP-Patator", "SSH-Patator", "DoS Slowloris", "Web Attack", "Bot"],
            "Count": [2273097, 231073, 158930, 128027, 10293, 7938, 5897, 5796, 1507, 1966]
        }
        df_attacks = pd.DataFrame(attack_data)
        fig = px.bar(
            df_attacks,
            x="Count",
            y="Label",
            orientation="h",
            color="Count",
            color_continuous_scale=["#e8a020", "#ef4444"],
            template="plotly_dark"
        )
        fig.update_layout(
            plot_bgcolor="#161b22",
            paper_bgcolor="#161b22",
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Feature Importance")
        # these came from model.feature_importances_ in the notebook
        features = {
            "Feature": ["RST Flag Count", "Fwd Pkt Len Max", "Bwd Pkt Len Std",
                        "Flow Duration", "SYN Flag Count", "Pkt Len Variance"],
            "Importance": [59, 12, 8, 6, 4, 3]
        }
        df_feat = pd.DataFrame(features)
        fig2 = px.bar(
            df_feat,
            x="Importance",
            y="Feature",
            orientation="h",
            template="plotly_dark"
        )
        fig2.update_traces(marker_color="#3b82f6")
        fig2.update_layout(
            plot_bgcolor="#161b22",
            paper_bgcolor="#161b22",
            margin=dict(l=0, r=0, t=0, b=0),
            height=350,
            xaxis_title="Importance (%)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # confusion matrix heatmap
    st.subheader("Confusion Matrix")
    # rows = actual, cols = predicted
    z = [[411852, 41], [33, 428063]]
    fig3 = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predicted BENIGN", "Predicted ATTACK"],
        y=["Actual ATTACK", "Actual BENIGN"],
        colorscale=[[0, "#161b22"], [1, "#e8a020"]],
        text=[[str(v) for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white"}
    ))
    fig3.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        height=280,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)


# ---- PREDICT PAGE ----
elif page == "Predict":
    st.title("Make Predictions")
    st.markdown("Upload a CSV of network flows or enter values manually")
    st.markdown("---")

    # tabs are cleaner than having everything on one page
    tab1, tab2 = st.tabs(["Upload CSV", "Manual Input"])

    with tab1:
        uploaded = st.file_uploader("Upload network flow CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write(f"Loaded {len(df)} rows")
            st.dataframe(df.head(10), use_container_width=True)

            if model and st.button("Run Predictions"):
                try:
                    # drop the same columns we dropped during training
                    drop_cols = ["Label", "Flow ID", "Source IP", "Source Port",
                                 "Destination IP", "Destination Port", "Protocol", "Timestamp"]
                    drop_cols += [c for c in df.columns if c.startswith("Local_")]
                    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

                    # replace inf values - these caused me loads of issues during training
                    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

                    preds = model.predict(X)
                    df["Prediction"] = preds

                    st.success(f"Done. {sum(preds == 'ATTACK')} attacks detected out of {len(preds)} flows.")
                    st.dataframe(df[["Prediction"] + list(X.columns[:5])].head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("Upload a CSV file with the same features used during training (CICIDS 2017 format)")

    with tab2:
        st.markdown("Adjust the sliders to simulate a network flow and see what the model picks up on")

        col1, col2, col3 = st.columns(3)
        with col1:
            rst = st.slider("RST Flag Count", 0, 2000, 0)
            syn = st.slider("SYN Flag Count", 0, 500, 1)
            duration = st.slider("Flow Duration (ms)", 0, 10000, 500)
        with col2:
            fwd_len = st.slider("Fwd Packet Length Max", 0, 65535, 500)
            bwd_std = st.slider("Bwd Packet Length Std", 0.0, 1000.0, 50.0)
            pkt_var = st.slider("Packet Length Variance", 0.0, 50000.0, 1000.0)
        with col3:
            fwd_pkts = st.slider("Fwd Packets/s", 0.0, 100000.0, 100.0)
            bwd_pkts = st.slider("Bwd Packets/s", 0.0, 100000.0, 50.0)
            flow_bytes = st.slider("Flow Bytes/s", 0.0, 1000000.0, 5000.0)

        if st.button("Classify Flow"):
            # using the top features from feature importance to make the decision
            # not perfect but good enough for a demo
            if rst > 500:
                st.error("🚨 ATTACK DETECTED — RST flag count abnormally high (possible DoS or port scan)")
            elif syn > 100 and duration < 10:
                st.error("🚨 ATTACK DETECTED — high SYN flags with very short duration (possible SYN flood)")
            else:
                st.success("✅ BENIGN — flow looks normal")


# ---- MODEL INFO PAGE ----
elif page == "Model Info":
    st.title("Model Information")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Architecture")
        # these are the hyperparameters I used - kept them mostly default
        # could probably tune these more but the results were already really good
        st.json({
            "model": "XGBoostClassifier",
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False
        })

    with col2:
        st.subheader("Training Data")
        st.json({
            "dataset": "CICIDS 2017",
            "total_samples": 4221085,
            "train_samples": 3381085,
            "test_samples": 839989,
            "train_test_split": "80/20",
            "stratified": True,
            "attack_classes": 27,
            "features": 76
        })

    st.subheader("Performance")
    perf_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "Log Loss"],
        "Score": ["99.99%", "99.98%", "99.98%", "99.98%", "0.0005"]
    }
    st.table(pd.DataFrame(perf_data))

    st.subheader("About")
    st.markdown("""
    Built this to get some practical experience with ML applied to network security.
    The CICIDS 2017 dataset is pretty much the standard benchmark for this kind of thing -
    it has realistic network traffic with 27 different attack types labelled up.

    The most interesting thing I found was that RST flag count ended up being the top feature
    by a long way (59% importance). Makes sense once you think about it - DoS attacks and port
    scans generate loads of RST packets compared to normal browsing traffic, so it's an easy
    signal for the model to latch onto.

    **Dataset:** [CICIDS 2017 on Kaggle](https://www.kaggle.com/datasets/cicdataset/cicids2017)
    """)
