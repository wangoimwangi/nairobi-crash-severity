# Accident Severity Classification System (ASCS)
### A Machine Learning Model for Traffic Accident Severity Classification to Support Emergency Dispatch in Nairobi

**MSc Information Technology Thesis — Strathmore University**
**Student:** Mary Wangoi Mwangi | 122174
**Supervisor:** Prof. Vincent Omwenga

---

## Overview

The Accident Severity Classification System (ASCS) is a machine learning-based decision-support tool developed to assist emergency dispatchers in Nairobi classify traffic accident severity in real time. The system classifies incidents as **HIGH severity** (requiring Advanced Life Support — ALS) or **LOW severity** (requiring Basic Life Support — BLS) based on dispatcher-entered situational variables from caller reports.

The system was developed as part of an MSc IT thesis at Strathmore University, using the RTA Addis Ababa Dataset (Bedane, 2020) as a validated High-Density East African Urban Proxy for Nairobi's emergency dispatch context.

---

## Live Demo

🔗 [https://nairobi-crash-severity.streamlit.app](https://nairobi-crash-severity.streamlit.app)

---

## Research Summary

| Item | Details |
|------|---------|
| Dataset | RTA Addis Ababa (Bedane, 2020) — Mendeley Data |
| Records | 12,316 records · 32 features · 2017–2020 |
| Champion Model | Balanced Random Forest |
| Threshold | F2-optimised at 0.40 |
| HIGH Severity Recall | 91.6% |
| Under-triage Rate | 8.4% |
| ROC-AUC | 0.698 |
| Domain Expert Score | 4.17 / 5 |

---

## System Architecture

The system takes 7 dispatcher inputs and automatically hydrates the remaining 21 features using a three-tier input architecture:

1. **Dispatcher inputs** — collision type, vehicle type, casualty count, vehicle count, cause of accident, location, and pedestrian involvement
2. **Auto-derived contextual features** — temporal features from system clock and weather from Open-Meteo API
3. **Modal defaults** — statistically justified defaults for low-impact variables

The 28-feature vector is passed to the Balanced Random Forest pipeline which returns HIGH or LOW with a confidence score, ALS/BLS recommendation, contributing risk factors, and nearest trauma centre.

---

## Repository Structure

nairobi-crash-severity/
│
├── app/
│   ├── app.py              # Main Streamlit application
│   ├── predictor.py        # Feature hydration and prediction logic
│   ├── hospitals.py        # Nairobi area mapping and hospital lookup
│   └── style.css           # UI styling
│
├── models/
│   ├── best_rf_compressed.pkl    # Trained Balanced RF model (compressed)
│   └── model_metadata.json       # Threshold and feature metadata
│
├── notebooks/
│   ├── 01_data_inspection_and_labeling.ipynb
│   ├── 02_model_training_and_evaluation.ipynb
│   ├── 03_statistical_testing_and_champion_identification.ipynb
│   └── 04_final_test_set_evaluation.ipynb
│
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
│
└── requirements.txt

---

## Model Development

Three classification pipelines were developed and compared:

| Model | Recall | F2 | ROC-AUC | Under-triage |
|-------|--------|----|---------|-------------|
| Logistic Regression (Baseline) | 0.891 | 0.467 | 0.587 | 10.9% |
| **Balanced Random Forest ★** | **0.916** | **0.523** | **0.698** | **8.4%** |
| XGBoost | 0.940 | 0.481 | 0.624 | 6.0% |

Champion selection was based on highest ROC-AUC and F2-Score, validated through Repeated Stratified K-Fold (5×3), Friedman test, and Wilcoxon signed-rank with Bonferroni correction (α=0.0167).

---

## Key Design Decisions

- **Safety-biased threshold (0.40):** The F2-optimised threshold weights recall twice over precision, reflecting the asymmetric cost of under-triage in emergency dispatch
- **ImbPipeline:** SMOTE applied only within training folds to prevent data leakage
- **Three-tier input architecture:** Minimises dispatcher cognitive load during high-pressure calls

---

## Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/wangoimwangi/nairobi-crash-severity.git
cd nairobi-crash-severity

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the application
cd app
streamlit run app.py
```

---

## Dataset Reference

Bedane, T. (2020). *Road traffic accident dataset of Addis Ababa city* [Data set]. Mendeley Data. https://doi.org/10.17632/xytv86278f.1

---

## License

This project is submitted in partial fulfilment of the requirements for the degree of Master of Science in Information Technology at Strathmore University. All rights reserved.

