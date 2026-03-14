"""
Threat Intel NLP — NVD Data Fetcher
Luke - CS @ Swansea Uni, second year

Pulls CVE records from the NVD API and saves them to a CSV for training.
NVD = National Vulnerability Database — maintained by NIST, free API, no key needed.
Each CVE has a description and a CVSS severity score (Critical/High/Medium/Low/None).

Docs: https://nvd.nist.gov/developers/vulnerabilities

Setup:
    pip install requests pandas tqdm
"""

import requests
import pandas as pd
import time
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_PATH = "./nvd_cves.csv"
TOTAL       = 20000    # how many CVEs to fetch — NVD has 200k+ but 20k is enough to train on
BATCH_SIZE  = 2000     # NVD API max per request is 2000
BASE_URL    = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# ── Fetch ─────────────────────────────────────────────────────────────────────
print(f"[*] Fetching {TOTAL} CVEs from NVD API...")

records = []
for start in tqdm(range(0, TOTAL, BATCH_SIZE)):
    params = {
        "startIndex": start,
        "resultsPerPage": BATCH_SIZE,
    }

    # NVD rate limits to 5 requests per 30 seconds without an API key
    # sleep 6 seconds between requests to stay under the limit
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    Error at index {start}: {e}")
        time.sleep(10)
        continue

    for item in data.get("vulnerabilities", []):
        cve = item.get("cve", {})

        # get english description
        descs = cve.get("descriptions", [])
        desc  = next((d["value"] for d in descs if d["lang"] == "en"), None)
        if not desc:
            continue

        # get CVSS severity — try v3.1 first, fall back to v3.0, then v2
        metrics    = cve.get("metrics", {})
        severity   = None
        cvss_score = None

        for version in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
            if version in metrics and metrics[version]:
                m = metrics[version][0]
                severity   = m.get("cvssData", {}).get("baseSeverity") or m.get("baseSeverity")
                cvss_score = m.get("cvssData", {}).get("baseScore")
                break

        if not severity:
            continue

        records.append({
            "cve_id":      cve.get("id", ""),
            "description": desc,
            "severity":    severity.upper(),   # CRITICAL, HIGH, MEDIUM, LOW
            "cvss_score":  cvss_score,
            "published":   cve.get("published", ""),
        })

    time.sleep(6)   # stay under rate limit

# ── Save ──────────────────────────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n[*] Saved {len(df)} CVEs → {OUTPUT_PATH}")
print(f"\n    Severity distribution:\n{df['severity'].value_counts()}")
