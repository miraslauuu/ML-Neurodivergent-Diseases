# ML for Neurodegenerative Diseases — Comparative Study

> **Goal:** build and compare ML models that predict patients’ cognitive status using demographic data, neuropsychological tests, and MRI volumetrics across several public datasets.

## TL;DR
- Unified preprocessing (imputation, one-hot, SMOTE).
- Strong baseline ensembles:
  - **Parkinson’s (UCI, voice):** H2O AutoML Stacked Ensemble, AUC ≈ **0.999**.
  - **Alzheimer’s (Kaggle, clinical):** Gradient Boosting, AUC ≈ **0.96**, F1 ≈ **0.91**.
  - **OASIS-2 (MRI + clinical):** Random Forest, AUC ≈ **0.99**, macro-F1 ≈ **0.93**.

---

## Repository structure
.</br>
├─ data/ </br>
│  ├─ Alzheimer/</br>
│  ├─ Dementia/</br>
│  └─ Parkinsons/</br>
├─ notebooks/</br>
│  ├─ dementia.ipynb</br>
│  ├─ alzheimer.ipynb</br>
│  └─ parkinsons.ipynb</br>
├─ README.md</br>
└─ ML_report_en.docx    # full report


---

## Datasets
| Dataset | Modality | Task | Size (approx.) |
|---|---|---|---|
| **OASIS-2 Dementia Prediction** | MRI volumetrics + clinical | 3-class (Nondemented / Converted / Demented) | 373 sessions / 150 persons |
| **Alzheimer’s (Kaggle)** | Clinical & demographic | Binary Dx | ~2,100 rows |
| **Parkinson’s (UCI)** | Voice features | Binary Dx | 197–756 rows (various variants) |
| **Parkinson’s Telemonitoring (UCI)** | Voice + time | Regression/derived classification | ~5,900 rows |

> Some sources require manual download/terms acceptance; see **ML_report_en.docx** for links and licenses.

---

## Methods (uniform pipeline)
- **Preprocessing:** median imputation (e.g., SES), removal of records lacking critical fields (e.g., MMSE when missing), one-hot encoding for categorical features, **SMOTE on the train split only**.
- **Models:** SVM, Random Forest, Gradient Boosting, Stacking (H2O AutoML).
- **Evaluation:** stratified 80/20 split, 5-fold CV on train; metrics include Accuracy, Precision, Recall, F1, AUC-ROC.
- **Feature selection:** RF importances / `SelectFromModel`; K-best where applicable.

---

## Results (high-level)
| Dataset | Best Model (key notes) | Result (test) |
|---|---|---|
| Parkinson’s (UCI) | H2O AutoML StackedEnsemble; SMOTE(k=2), 600s | **AUC ≈ 0.999**, Acc ≈ 96.7%, Recall 1.00 |
| Alzheimer’s (Kaggle) | GradientBoosting; SelectKBest(k=13); scaled | **AUC ≈ 0.96**, Acc ≈ 0.91, F1 ≈ 0.91 |
| OASIS-2 | RandomForest (200 trees, `class_weight='balanced'`); SMOTE(k=5) | **AUC ≈ 0.99**, Acc/Prec/Rec/F1 ≈ 0.93 |

> On OASIS-2, reducing to **four** key attributes (MR Delay, SES, MMSE, CDR) preserved most performance while improving interpretability.

---

## Quick start

### 1) Environment
bash
# Python >= 3.10 recommended
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -U pip

# Core deps (adjust as needed)
pip install numpy pandas scikit-learn imbalanced-learn jupyter matplotlib h2o

---
## Notes & Decisions
- **Why ensembles?** They consistently outperformed simpler baselines across heterogeneous datasets.
- **SMOTE scope:** applied **only on training folds** to avoid leakage; `k` tuned per dataset size.
- **Interpretability:** OASIS-2 works well with just **4** clinically meaningful features.
- **Limitations:** relatively small cohorts; recommend external-cohort validation and longitudinal stability tests before clinical use.

## How to cite
If you use this repository or the attached report in academic work, please cite:

> Alkhovik, M. (2025). *Comparative analysis of machine learning methods for the classification of neurodegenerative diseases.* Course: Introduction to Machine Learning. (See `ML_report_en.docx`).

## Licenses
- **Code:** MIT (add a `LICENSE` file).
- **Data:** obey each dataset’s original license/terms (e.g., CC BY / CC BY-NC). See the report for details.

## Acknowledgements
Thanks to the custodians of OASIS, the UCI Machine Learning Repository, and Kaggle for providing open datasets that make this research possible.
