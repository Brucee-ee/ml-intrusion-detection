# ML Intrusion Detection System

A machine learning based network intrusion detection system trained on real network traffic data. Built as a personal project to explore the intersection of ML and cybersecurity.

## What it does

Takes network flow data and classifies it as either benign or an attack. Trained on the CICIDS 2017 dataset which contains over 4 million real network flows covering 27 different attack types including DoS, DDoS, port scans, brute force, and web attacks.

The model runs as a Streamlit web app with three pages:
- **Dashboard** overview of model performance and attack distribution
- **Predict** upload a CSV of network flows and get predictions back, or manually adjust features to test the classifier
- **Model Info** architecture details and training stats

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 0.9999 |
| F1 Score | 0.9998 |
| Precision | 0.9998 |
| Recall | 0.9998 |
| Log Loss | 0.0005 |

Out of 839,989 test samples, 41 false positives and 33 false negatives. RST Flag Count was the most important feature at around 59% importance, which makes sense given how heavily DoS and port scan attacks rely on resetting connections.

## Tech stack

- Python 3.13
- XGBoost
- scikit-learn
- Streamlit
- Plotly
- pandas / numpy

## Dataset

CICIDS 2017 from the Canadian Institute for Cybersecurity.
4,199,942 rows, 85 features, 27 attack categories.
Available on Kaggle, search "CICIDS 2017".

## Running it locally

```bash
git clone https://github.com/yourusername/ml-intrusion-detection
cd ml-intrusion-detection

python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

Download the CICIDS 2017 dataset from Kaggle and place the CSVs in the `data/` folder, then run the notebook to train and save the model:

```bash
jupyter notebook notebooks/explore.ipynb
```

Once the model is saved in `model/`, start the app:

```bash
streamlit run app.py
```

## Project structure

```
ml-intrusion-detection/
├── data/               # CICIDS 2017 CSVs (not included, download from Kaggle)
├── model/              # saved XGBoost model + confusion matrix
├── notebooks/
│   └── explore.ipynb   # data cleaning, training, evaluation
├── app.py              # Streamlit dashboard
├── requirements.txt
└── README.md
```

## What I learned

Getting near-perfect accuracy wasn't a surprise once I looked at the feature importances. TCP flag patterns are strong indicators of malicious traffic. The more interesting part was understanding why the model works, not just that it does. RST flags dominate because attackers can't really hide the connection patterns that scanning and flooding produce at the network level.

If I were to extend this I'd look at training on more recent datasets and testing how the model handles adversarial traffic designed to evade detection.

## Author

Luke, CS student at Swansea University  
[github.com/yourusername](https://github.com/yourusername)
