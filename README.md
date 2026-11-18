# Explainable Machine Learning for Predicting Appointment No-Show Behaviour

This project builds and evaluates an explainable machine-learning system that predicts whether a scheduled medical appointment will end in a no-show. It then compares two explanation methods, SHAP and LIME, on the same predictions. It supports a dissertation on behavioural risk relevant to surgical cancellation during pre-operative assessment. No-show is used as a behavioural proxy. The system does not predict surgical cancellation and is not a clinical tool.

**Data:** the public Brazil medical appointment dataset (Kaggle, "Medical Appointment No Shows", May 2016, about 110,500 cleaned records). The CSV is not included: download `KaggleV2-May-2016.csv` from Kaggle.

**Pipeline:** a ten-layer, leakage-controlled workflow:
- Data cleaning.
- Feature engineering: lead time, weekday, previous no-show history (earlier appointments only), age bands and comorbidity count.
- A chronological 70/15/15 split.
- Imbalance handling: original distribution, class weights and SMOTE-NC.
- Five models: Logistic Regression, linear SVM, Random Forest, XGBoost and a time-aware stacked ensemble.
- Threshold selection on validation data.
- One final test on future appointments.

**Results:** tuned XGBoost was selected, because the ensemble did not beat it by the preset margin. On the future test period it reached ROC-AUC 0.741 and PR-AUC 0.343, with recall 80.8% and precision 28.9% at the F1-optimal threshold. Lead time, age and previous no-show rate were the strongest predictors.

**SHAP vs LIME (300 test cases):** the methods agreed on the direction of effects 91% of the time but chose the same top feature in only 31% of cases. SHAP was exact, deterministic and about 400 times faster. It needed about 4 features to reach 80% of an explanation, against about 11 for LIME, whose local fit was modest (mean R² 0.35).

## Files
| File | What it is |
|---|---|
| `noshow_pipeline.ipynb` | Full analysis notebook (Google Colab or Jupyter) |
| `noshow_pipeline.py` | Same code as a standalone script |
| `app/index.html` | Browser app: enter an appointment, get the risk, SHAP and LIME-style explanations (model built in; open the file in a browser) |
| `docs/System_Architecture.docx` | Ten-layer system architecture |

## Run it
1. Install Python 3.11 to 3.13, then `pip install -r requirements.txt`.
2. Download `KaggleV2-May-2016.csv` and put it next to the script (or in a `data` folder). In Colab you can keep it in Google Drive instead.
3. Run `python noshow_pipeline.py`, or open the notebook and choose Run all. Figures are saved to `figures/`.

## Limitations
One city, about six weeks of 2016 data, limited predictors, and no user study of how useful the explanations are. Outputs are analytical evidence, not clinical decisions.
