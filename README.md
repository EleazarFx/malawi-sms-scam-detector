# Malawi SMS Scam Detector

> A machine learning system for detecting SMS scams in **Chichewa** and **Malawian English**, tuned for the local mobile money fraud landscape (Airtel Money, TNM Mpamba, fake NGO relief, "miracle money" cults, and transporter/parcel scams).

---

## Overview

Mobile phone scams are one of the fastest-growing forms of fraud in Malawi. Every day, thousands of Malawians receive SMS messages claiming they have:

- Won a prize from Premier Bet, Premier Lotto, or FESA
- Been selected to receive **MK 75,000** from "JB Foundation / Mtukula Pakhomo"
- A parcel held at a border ("ndine transporter, katundu wa m'bale wanu…") that requires a transport fee
- A blocked Airtel Money / TNM Mpamba account that must be "unblocked" by calling a number
- An invitation to a "Miracle Money" satanic church promising millions

Most of these messages arrive in **Chichewa**, **English**, or **mixed Chichewa-English** — and no publicly available scam-detection dataset exists for this context.

This project builds one — and trains a supervised classifier on it.

---

## Demo

<!-- Screenshot the FastAPI page and drop it here -->

![Demo screenshot](docs/demo.png)

Paste an SMS, get a verdict and a scam probability:



---

## Dataset

**678 SMS messages** (339 scam, 339 legitimate) collected and hand-labeled for the Malawian context.

| Class | Examples |
|-------|----------|
| **Scam** | Fake mobile money agents, fake NGO/government relief, transporter scams, "miracle money" cults, fake lotteries, SIM-block phishing |
| **Legitimate** | Personal Chichewa conversation, Airtel/TNM service alerts, Ministry of Health campaigns, bank transaction notifications, church announcements |

### Composition

- **Languages:** Chichewa, English, mixed code-switching
- **Labeling rule:** A message is `scam` if it asks the recipient to **call a number, send money, click a link, or share a PIN/password**. Otherwise it's `legitimate`.
- **Collection:** Manually curated from public scam reports, personal inboxes (anonymized), and simulated variants of common scam templates.
- **Ethics:** All phone numbers, names, and personal identifiers have been fictionalized. No real personal data is included.

### Why this matters

There is **no public Chichewa SMS scam dataset**. This is the first of its kind.

---

## How It Works

### 1. Preprocessing (`src/preprocess.py`)
- Lowercase normalization
- URLs, phone numbers, and money amounts replaced with stable tokens (`url`, `phone`, `money`) so the model learns *patterns*, not specific digits
- Punctuation stripped

### 2. Feature Engineering (`src/features.py` + `src/rules.py`)
Two complementary feature types combined into one vector:

**a) TF-IDF over cleaned text**
- 1-2 word n-grams, up to 5,000 features
- Custom Chichewa + English stopword list (removes `ku`, `iyi`, `wa`, `kuti` …)
- Sublinear term frequency

**b) Handcrafted rule features**
Domain knowledge encoded as binary/numeric signals:
- `agent_scam` — transporter / "m'bale wanu" parcel scam markers
- `miracle_money` — satanic church / miracle money markers
- `money_promise` — "you won", "mwapambana", "mphoto"
- `lottery` — Premier Bet, Premier Lotto, FESA
- `relief_scam` — DODMA / UNICEF / COVID / cyclone
- `urgency` — "mwachangu", "musachedwe"
- `phone_count`, `length_scaled`

### 3. Classifier
**Linear SVM** with class weights balanced. Chosen over Naive Bayes and Logistic Regression via 5-fold cross-validation.

---

## Results

Trained on 542 messages, evaluated on a held-out 136-message test set.

| Metric | Score |
|--------|-------|
| **Cross-val F1 (5-fold)** | **0.957 ± 0.028** |
| Test accuracy | 0.96 |
| **Scam precision** | **0.94** |
| **Scam recall** | **0.97** |
| Ham precision | 0.97 |
| Ham recall | 0.94 |

### Confusion matrix (threshold = 0.5)

### Threshold tuning

We deploy at **threshold = 0.4**, not 0.5, because **missing a scam (false negative) is far more costly than a false alarm (false positive)**. A missed scam means real money lost. A false alarm just shows a warning the user can ignore.

| Threshold | FP | FN | When to use |
|-----------|----|----|-------------|
| 0.3 | 41 | 0 | Very aggressive — flag lots of borderline messages |
| **0.4** | **7** | **0** | ⭐ **Production choice — catch every scam** |
| 0.5 | 4 | 2 | Balanced F1 — misses 2 scams |
| 0.6 | 1 | 9 | Conservative — misses 9 scams |

---

## Quick Start

### 1. Clone and set up

```bash
git clone <your-repo-url>
cd malawi-sms-scam-detector
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt

#Train model
python src/train.py

#Outputs:
models/scam_detector.pkl — the trained pipeline

models/score_distribution.png — plot of predicted probabilities by class

#RUN THE WEB APP
uvicorn app.main:app --reload

#Use the JSON API
curl "http://127.0.0.1:8000/api/predict?text=Mwapambana%20K500000"

#json output:
{"label": "scam", "confidence": 0.4328, "threshold": 0.4}

#Projec Structure
malawi-sms-scam-detector/
├── app/
│   ├── main.py                  FastAPI server
│   └── templates/
│       └── index.html           Web UI
├── data/
│   ├── raw/
│   │   └── sms_dataset.csv      Original hand-labeled data
│   └── interim/
│       └── sms_deduped.csv      After deduplication
├── models/
│   ├── scam_detector.pkl        Trained pipeline (pickle)
│   └── score_distribution.png   Test-set score plot
├── notebooks/
│   └── 01_data_audit.ipynb      Duplicate/label auditing
├── src/
│   ├── clean_data.py            Deduplication + filtering
│   ├── preprocess.py            Text normalization
│   ├── rules.py                 Handcrafted features
│   ├── features.py              TF-IDF + TextRouter
│   ├── train.py                 Training + evaluation
│   └── predict.py               Inference (CLI + importable)
├── requirements.txt
└── README.md

#LIMITATIONS AND ETHICS
It does not block messages — it only flags them.

It cannot verify whether a specific phone number is real.

It does not understand conversation history or sender reputation.

It will occasionally flag legitimate messages that use scam-shaped language (e.g., a real transporter asking for a ride from Mwanza border).

#Future Work
□ Expand dataset to 2,000+ messages with more hard examples (short scams, mixed-language scams).
□ Multilingual transformer model (XLM-R, AfriBERTa) fine-tuned on this data.
□ Active learning — user-reported scam SMS feeds back into training.
□ Rule-based safety net — regex overrides for known high-confidence patterns.
□ USSD / Android SMS integration — deploy directly on a phone or telco SMS gateway.
□ Per-sender reputation — cluster messages by sender and flag repeat offenders.
□ Explainability UI — show which words triggered the scam verdict, so users learn.


#Requirements
Python 3.10+

scikit-learn, pandas, numpy

matplotlib, seaborn (for EDA)

joblib (model persistence)

fastapi, uvicorn, jinja2 (for the web app)



==============================================================
AUTHOR
Built by Eleazar Toto - eleazarctoto@gmail.com
==============================================================
License
MIT — free to use, modify, and redistribute with attribution.