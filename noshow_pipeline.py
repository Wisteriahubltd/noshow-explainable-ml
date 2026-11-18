#!/usr/bin/env python3
"""Explainable ML for appointment no-show prediction (SHAP vs LIME) - standalone script.

Same code as noshow_pipeline.ipynb, runnable outside Colab:
    pip install pandas numpy scikit-learn xgboost shap lime imbalanced-learn matplotlib seaborn joblib scipy
    python noshow_pipeline.py
Put KaggleV2-May-2016.csv next to this script (or in a data/ folder). Figures are saved to ./figures and
the web-app model to noshow_model.json. In Colab, use the notebook instead.
"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs('figures', exist_ok=True)
_fig_no = [0]
def _save_show(*a, **k):
    _fig_no[0] += 1
    plt.savefig('figures/fig_%03d.png' % _fig_no[0], dpi=150, bbox_inches='tight'); plt.close('all')
plt.show = _save_show
def display(x): print(x)


# ===== notebook cell 2 =====
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import warnings; warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
RANDOM_STATE = 42

# ===== notebook cell 3 =====
# 1.2 Load the data. One cell, no choices to make: it uses the first source that works.
#   1) the CSV already in this Colab session (or next to this script)  ->  2) Google Drive  ->  3) a one-time upload prompt
import os, glob, shutil
CSV_NAME   = 'KaggleV2-May-2016.csv'
DRIVE_PATH = '/content/drive/MyDrive/noshow_project/' + CSV_NAME      # edit only if you keep the file elsewhere in Drive

try:
    import google.colab                      # True only inside Colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def find_local():
    # also catches names like 'KaggleV2-May-2016 (1).csv' that Colab creates when you upload the same file twice
    for pattern in (CSV_NAME, '*May-2016*.csv', '/content/*May-2016*.csv', 'data/*May-2016*.csv'):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None

csv_path = find_local()

if csv_path is None and IN_COLAB:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        if os.path.exists(DRIVE_PATH):
            csv_path = DRIVE_PATH
        else:
            hits = glob.glob('/content/drive/MyDrive/**/*May-2016*.csv', recursive=True)
            csv_path = hits[0] if hits else None
    except Exception as err:
        print('Google Drive not available (' + type(err).__name__ + '); falling back to upload.')

if csv_path is None and IN_COLAB:
    from google.colab import files
    print('Choose KaggleV2-May-2016.csv and wait until the upload bar reaches 100% ...')
    up = files.upload()
    if not up:
        raise FileNotFoundError('No file was uploaded. Run this cell again and select KaggleV2-May-2016.csv.')
    name = next(iter(up))
    with open(CSV_NAME, 'wb') as fh:          # keep a copy so re-running never asks again
        fh.write(up[name])
    csv_path = CSV_NAME

if csv_path is None:
    raise FileNotFoundError('Could not find ' + CSV_NAME + '. Put it next to this script, in a "data" folder, or in Google Drive at ' + DRIVE_PATH)

df_raw = pd.read_csv(csv_path)
need = {'PatientId', 'AppointmentID', 'Gender', 'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood',
        'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'No-show'}
missing = need - set(df_raw.columns)
if missing:
    raise ValueError('This does not look like KaggleV2-May-2016.csv. Missing columns: ' + ', '.join(sorted(missing)))
print('Loaded', df_raw.shape[0], 'rows and', df_raw.shape[1], 'columns from', csv_path)
df_raw.head()

# ===== notebook cell 5 =====
print('Shape:', df_raw.shape)
df_raw.info()
print('\nMissing values:\n', df_raw.isna().sum())
print('\nExact duplicate rows:', df_raw.duplicated().sum())
print('Duplicate AppointmentIDs:', df_raw['AppointmentID'].duplicated().sum())
print('Unique patients:', df_raw['PatientId'].nunique())
print('\nAge summary:\n', df_raw['Age'].describe())
print('Age < 0:', (df_raw['Age'] < 0).sum(), '| Age > 100:', (df_raw['Age'] > 100).sum())
print('\nTarget distribution:\n', df_raw['No-show'].value_counts(normalize=True).round(4))

# ===== notebook cell 6 =====
df = df_raw.copy()

# Fix misspelled / awkward column names
df = df.rename(columns={'Hipertension':'Hypertension','Handcap':'Handicap',
                        'Neighbourhood':'Neighborhood','No-show':'NoShow',
                        'SMS_received':'SMSReceived'})

# Target: 1 = no-show, 0 = attended
df['Y'] = (df['NoShow'] == 'Yes').astype(int)

# Dates
df['ScheduledDay']   = pd.to_datetime(df['ScheduledDay']).dt.tz_localize(None)
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.tz_localize(None)

# Lead time in whole days (AppointmentDay has no time-of-day, so compare dates)
df['LeadTime'] = (df['AppointmentDay'].dt.normalize() - df['ScheduledDay'].dt.normalize()).dt.days

log = {'start_rows': len(df)}
df = df.drop_duplicates();               log['after_duplicates'] = len(df)
df = df[df['Age'] >= 0];                 log['after_negative_age'] = len(df)
df = df[df['LeadTime'] >= 0];            log['after_negative_leadtime'] = len(df)

# Chronological order (needed for history features and the temporal split)
df = df.sort_values(['AppointmentDay','ScheduledDay','AppointmentID']).reset_index(drop=True)

print(pd.Series(log))
print('\nNo-show rate after cleaning: %.4f' % df['Y'].mean())
print('Date range:', df['AppointmentDay'].min().date(), 'to', df['AppointmentDay'].max().date())
print('\nHandicap values:', sorted(df['Handicap'].unique()))
df.head()

# ===== notebook cell 9 =====
# 3.1 Previous attendance history (past appointments only)
g = (df.groupby(['PatientId','AppointmentDay'])
       .agg(n=('Y','size'), s=('Y','sum')).reset_index()
       .sort_values(['PatientId','AppointmentDay']))
g['PrevAppts']   = g.groupby('PatientId')['n'].cumsum() - g['n']
g['PrevNoShows'] = g.groupby('PatientId')['s'].cumsum() - g['s']

df = df.merge(g[['PatientId','AppointmentDay','PrevAppts','PrevNoShows']],
              on=['PatientId','AppointmentDay'], how='left')
df['HasPreviousHistory'] = (df['PrevAppts'] > 0).astype(int)
df['PrevNoShowRate'] = np.where(df['PrevAppts'] > 0, df['PrevNoShows'] / df['PrevAppts'], 0.0)

# Leakage sanity check: a patient's earliest appointment date must have no history
first_day = df.groupby('PatientId')['AppointmentDay'].transform('min')
assert (df.loc[df['AppointmentDay'] == first_day, 'PrevAppts'] == 0).all()
print('History leakage check passed')
print(df['HasPreviousHistory'].value_counts(normalize=True).round(4))

# ===== notebook cell 10 =====
# 3.2 Calendar features
df['ApptDayOfWeek']  = df['AppointmentDay'].dt.day_name()
df['SchedDayOfWeek'] = df['ScheduledDay'].dt.day_name()

# 3.3 Age bands (interpretability groupings, not clinical thresholds)
df['AgeBand'] = pd.cut(df['Age'], bins=[-1,17,29,44,59,74,200],
                       labels=['0-17','18-29','30-44','45-59','60-74','75+'])

# 3.4 Comorbidity count (dataset-specific proxy; Handicap treated as present/absent)
df['ComorbidityCount'] = (df['Hypertension'] + df['Diabetes'] + df['Alcoholism']
                          + (df['Handicap'] > 0).astype(int))

engineered = ['LeadTime','PrevAppts','PrevNoShows','PrevNoShowRate','HasPreviousHistory',
              'ApptDayOfWeek','SchedDayOfWeek','AgeBand','ComorbidityCount']
df[engineered + ['Y']].describe(include='all').T

# ===== notebook cell 11 =====
# 3.5 Quick look: does no-show rate vary with the new features?
df['LeadBin'] = pd.cut(df['LeadTime'], [-1,0,3,7,14,30,400],
                       labels=['same day','1-3','4-7','8-14','15-30','31+'])
for col in ['LeadBin','HasPreviousHistory','AgeBand','ComorbidityCount','SMSReceived']:
    print(f'\nNo-show rate by {col}')
    print(df.groupby(col)['Y'].agg(['mean','size']).round(4))

fig, ax = plt.subplots(1, 3, figsize=(15,4))
df.groupby('LeadBin')['Y'].mean().plot.bar(ax=ax[0], title='No-show rate by lead time')
df.groupby('AgeBand')['Y'].mean().plot.bar(ax=ax[1], title='No-show rate by age band')
df.groupby('PrevNoShows')['Y'].mean().head(6).plot.bar(ax=ax[2], title='No-show rate by prior no-shows')
plt.tight_layout(); plt.show()

# ===== notebook cell 12 =====
# Save progress so a Colab disconnect does not cost you the work
df.to_csv('noshow_features.csv', index=False)
print('Saved noshow_features.csv  | rows:', len(df), '| columns:', df.shape[1])

# ===== notebook cell 15 =====
# 4.1 Chronological 70 / 15 / 15 split on whole appointment days
day_share = df.groupby('AppointmentDay').size().cumsum() / len(df)
train_end = day_share[day_share >= 0.70].index[0]
val_end   = day_share[day_share >= 0.85].index[0]

train = df[df['AppointmentDay'] <= train_end].copy()
val   = df[(df['AppointmentDay'] > train_end) & (df['AppointmentDay'] <= val_end)].copy()
test  = df[df['AppointmentDay'] > val_end].copy()

summary = pd.DataFrame({
    'rows':      [len(train), len(val), len(test)],
    'share':     [len(train)/len(df), len(val)/len(df), len(test)/len(df)],
    'from':      [d['AppointmentDay'].min().date() for d in (train, val, test)],
    'to':        [d['AppointmentDay'].max().date() for d in (train, val, test)],
    'noshow_rate': [d['Y'].mean() for d in (train, val, test)]},
    index=['train','validation','test']).round(4)
print(summary)
assert train['AppointmentDay'].max() < val['AppointmentDay'].min() < test['AppointmentDay'].min()
print('Temporal ordering check passed')

# ===== notebook cell 16 =====
# 4.2 Encoding and scaling (fitted on TRAIN only)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def add_model_columns(d, freq_map):
    d = d.copy()
    d['GenderMale']   = (d['Gender'] == 'M').astype(int)
    d['HandicapFlag'] = (d['Handicap'] > 0).astype(int)
    d['NeighborhoodFreq'] = d['Neighborhood'].map(freq_map).fillna(0.0)  # unseen -> 0
    return d

# Neighborhood has ~80 levels: use its share of TRAIN appointments (no target information used)
freq_map = train['Neighborhood'].value_counts(normalize=True).to_dict()
train_m, val_m, test_m = [add_model_columns(d, freq_map) for d in (train, val, test)]

numeric_cols = ['Age','LeadTime','PrevAppts','PrevNoShows','PrevNoShowRate',
                'ComorbidityCount','NeighborhoodFreq']
binary_cols  = ['GenderMale','Scholarship','Hypertension','Diabetes','Alcoholism',
                'HandicapFlag','SMSReceived','HasPreviousHistory']
cat_cols     = ['ApptDayOfWeek','SchedDayOfWeek','AgeBand']

preprocess = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('bin', 'passthrough', binary_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)],
    verbose_feature_names_out=False)

preprocess.fit(train_m)
feature_names = list(preprocess.get_feature_names_out())

def transform(d):
    return pd.DataFrame(preprocess.transform(d), columns=feature_names, index=d.index)

X_train, X_val, X_test = transform(train_m), transform(val_m), transform(test_m)
y_train, y_val, y_test = train['Y'].values, val['Y'].values, test['Y'].values

print('Feature matrix shapes:', X_train.shape, X_val.shape, X_test.shape)
print('Number of features:', len(feature_names))

# ===== notebook cell 17 =====
# 4.3 Class-imbalance strategies (applied to TRAINING data only)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTENC

# Strategy B: class weights  w = N / (K * N_c)
classes = np.array([0, 1])
w = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {int(c): float(x) for c, x in zip(classes, w)}
print('Class weights:', {k: round(v, 3) for k, v in class_weight_dict.items()})

# Strategy A: SMOTE-NC (treats binary / one-hot columns as categorical, not interpolated)
cat_idx = [i for i, c in enumerate(X_train.columns) if c not in numeric_cols]
smote = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print('\nTraining class counts')
print(' original :', dict(zip(*np.unique(y_train, return_counts=True))))
print(' SMOTE    :', dict(zip(*np.unique(y_train_sm, return_counts=True))))
print('Test set keeps its natural class balance:', dict(zip(*np.unique(y_test, return_counts=True))))

# ===== notebook cell 18 =====
# 4.4 Save everything so later sections can restart from here
import pickle
with open('splits.pkl', 'wb') as f:
    pickle.dump(dict(X_train=X_train, X_val=X_val, X_test=X_test,
                     y_train=y_train, y_val=y_val, y_test=y_test,
                     X_train_sm=X_train_sm, y_train_sm=y_train_sm,
                     class_weight_dict=class_weight_dict,
                     feature_names=feature_names, numeric_cols=numeric_cols), f)
print('Saved splits.pkl')

# ===== notebook cell 21 =====
# 5.0 Reload saved splits if the Colab session was reset
try:
    X_train
except NameError:
    import pickle
    globals().update(pickle.load(open('splits.pkl', 'rb')))
    RANDOM_STATE = 42
    print('Reloaded splits.pkl')

import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             confusion_matrix, brier_score_loss)

def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.nanargmax(f1[:-1])])

def evaluate(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn)
    return dict(ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                Precision=prec, Recall=rec, F1=2*prec*rec/(prec+rec+1e-12),
                Specificity=tn/(tn+fp), Brier=brier_score_loss(y, p), Threshold=thr)

# ===== notebook cell 22 =====
# 5.1 Model definitions (first-pass settings; tuning comes in Section 6)
def make_model(name, strategy):
    cw  = class_weight_dict if strategy == 'class_weight' else None
    spw = class_weight_dict[1] / class_weight_dict[0] if strategy == 'class_weight' else 1.0
    if name == 'Logistic Regression':
        return LogisticRegression(max_iter=2000, class_weight=cw, random_state=RANDOM_STATE)
    if name == 'SVM (linear, calibrated)':
        return CalibratedClassifierCV(LinearSVC(C=1.0, class_weight=cw, max_iter=5000,
                                                random_state=RANDOM_STATE), method='sigmoid', cv=3)
    if name == 'Random Forest':
        return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                      class_weight=cw, n_jobs=-1, random_state=RANDOM_STATE)
    if name == 'XGBoost':
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=spw, tree_method='hist',
                             eval_metric='logloss', n_jobs=-1, random_state=RANDOM_STATE)

def training_data(strategy):
    return (X_train_sm, y_train_sm) if strategy == 'smote' else (X_train, y_train)

model_names = ['Logistic Regression', 'SVM (linear, calibrated)', 'Random Forest', 'XGBoost']
strategies  = ['original', 'class_weight', 'smote']

# ===== notebook cell 23 =====
# 5.2 Train everything and score on the validation set (takes several minutes)
fitted, val_probs, rows = {}, {}, []
for strat in strategies:
    Xt, yt = training_data(strat)
    for name in model_names:
        t0 = time.time()
        m = make_model(name, strat).fit(Xt, yt)
        p = m.predict_proba(X_val)[:, 1]
        thr = best_f1_threshold(y_val, p)          # threshold chosen on VALIDATION
        rows.append(dict(Model=name, Strategy=strat, **evaluate(y_val, p, thr)))
        fitted[(name, strat)], val_probs[(name, strat)] = m, p
        print(f'{name:26s} | {strat:12s} | ROC-AUC {rows[-1]["ROC_AUC"]:.4f} | {time.time()-t0:5.0f}s')

results = pd.DataFrame(rows)

# ===== notebook cell 24 =====
# 5.3 Comparison tables
show = results.sort_values('PR_AUC', ascending=False).round(4).reset_index(drop=True)
display(show)

print('\nROC-AUC by model and strategy')
display(results.pivot(index='Model', columns='Strategy', values='ROC_AUC').round(4))
print('PR-AUC by model and strategy')
display(results.pivot(index='Model', columns='Strategy', values='PR_AUC').round(4))

print('Reference: a model with no skill has PR-AUC about', round(y_val.mean(), 4))

# ===== notebook cell 25 =====
# 5.4 Save results
import pickle
results.to_csv('baseline_results.csv', index=False)
with open('val_probs.pkl', 'wb') as f:
    pickle.dump(val_probs, f)
print('Saved baseline_results.csv and val_probs.pkl')

# ===== notebook cell 28 =====
# 6.1 Random search: XGBoost (25 configurations, a few minutes)
STRATEGY = 'original'          # 'original' or 'class_weight'
cw  = class_weight_dict if STRATEGY == 'class_weight' else None
spw = class_weight_dict[1] / class_weight_dict[0] if STRATEGY == 'class_weight' else 1.0
rng = np.random.default_rng(RANDOM_STATE)

def sample_xgb():
    return dict(n_estimators=int(rng.choice([200, 400, 600, 800])),
                learning_rate=float(rng.choice([0.02, 0.03, 0.05, 0.08])),
                max_depth=int(rng.choice([3, 4, 5, 6, 8])),
                min_child_weight=int(rng.choice([1, 5, 10, 20, 50])),
                subsample=float(rng.choice([0.6, 0.8, 1.0])),
                colsample_bytree=float(rng.choice([0.5, 0.7, 0.9, 1.0])),
                reg_lambda=float(rng.choice([1, 5, 10, 30])),
                gamma=float(rng.choice([0, 0.5, 1, 3])))

xgb_rows = []
for i in range(25):
    p = sample_xgb(); t0 = time.time()
    m = XGBClassifier(**p, scale_pos_weight=spw, tree_method='hist', eval_metric='logloss',
                      n_jobs=-1, random_state=RANDOM_STATE).fit(X_train, y_train)
    pv = m.predict_proba(X_val)[:, 1]
    xgb_rows.append(dict(**p, ROC_AUC=roc_auc_score(y_val, pv), PR_AUC=average_precision_score(y_val, pv)))
    print(f'XGB {i+1:2d}/25  ROC-AUC {xgb_rows[-1]["ROC_AUC"]:.4f}  PR-AUC {xgb_rows[-1]["PR_AUC"]:.4f}  ({time.time()-t0:.0f}s)')

xgb_search = pd.DataFrame(xgb_rows).sort_values('PR_AUC', ascending=False).reset_index(drop=True)
display(xgb_search.head(5).round(4))

# ===== notebook cell 29 =====
# 6.2 Random search: Random Forest (10 configurations, ~5-10 minutes)
def sample_rf():
    mf = rng.choice(['sqrt', '0.5', '0.7'])
    return dict(n_estimators=int(rng.choice([200, 300])),
                max_depth=int(rng.choice([8, 12, 16, 20])),
                min_samples_leaf=int(rng.choice([5, 10, 20, 50])),
                max_features=mf if mf == 'sqrt' else float(mf))

rf_rows = []
for i in range(10):
    p = sample_rf(); t0 = time.time()
    m = RandomForestClassifier(**p, class_weight=cw, n_jobs=-1,
                               random_state=RANDOM_STATE).fit(X_train, y_train)
    pv = m.predict_proba(X_val)[:, 1]
    rf_rows.append(dict(**p, ROC_AUC=roc_auc_score(y_val, pv), PR_AUC=average_precision_score(y_val, pv)))
    print(f'RF {i+1:2d}/10  ROC-AUC {rf_rows[-1]["ROC_AUC"]:.4f}  PR-AUC {rf_rows[-1]["PR_AUC"]:.4f}  ({time.time()-t0:.0f}s)')

rf_search = pd.DataFrame(rf_rows).sort_values('PR_AUC', ascending=False).reset_index(drop=True)
display(rf_search.head(5).round(4))

# ===== notebook cell 30 =====
# 6.3 Linear baselines: tune the regularisation strength C
lin_rows = []
for C in [0.01, 0.1, 1, 10]:
    lr = LogisticRegression(C=C, max_iter=2000, class_weight=cw,
                            random_state=RANDOM_STATE).fit(X_train, y_train)
    pv = lr.predict_proba(X_val)[:, 1]
    lin_rows.append(dict(Model='Logistic Regression', C=C, ROC_AUC=roc_auc_score(y_val, pv),
                         PR_AUC=average_precision_score(y_val, pv)))
for C in [0.01, 0.1, 1]:
    sv = CalibratedClassifierCV(LinearSVC(C=C, class_weight=cw, max_iter=5000,
                                          random_state=RANDOM_STATE), method='sigmoid', cv=3).fit(X_train, y_train)
    pv = sv.predict_proba(X_val)[:, 1]
    lin_rows.append(dict(Model='SVM (linear, calibrated)', C=C, ROC_AUC=roc_auc_score(y_val, pv),
                         PR_AUC=average_precision_score(y_val, pv)))
lin_search = pd.DataFrame(lin_rows)
display(lin_search.round(4))

# ===== notebook cell 31 =====
# 6.4 Refit the best configuration of each model, compare with the baseline, and check overfitting
best = {
    'XGBoost': {k: xgb_search.loc[0, k] for k in
                ['n_estimators','learning_rate','max_depth','min_child_weight','subsample',
                 'colsample_bytree','reg_lambda','gamma']},
    'Random Forest': {k: rf_search.loc[0, k] for k in
                      ['n_estimators','max_depth','min_samples_leaf','max_features']},
}
for k in ['n_estimators','max_depth','min_child_weight']:
    if k in best['XGBoost']: best['XGBoost'][k] = int(best['XGBoost'][k])
best['Random Forest']['n_estimators']      = int(best['Random Forest']['n_estimators'])
best['Random Forest']['max_depth']         = int(best['Random Forest']['max_depth'])
best['Random Forest']['min_samples_leaf']  = int(best['Random Forest']['min_samples_leaf'])
mf = best['Random Forest']['max_features']
best['Random Forest']['max_features'] = mf if isinstance(mf, str) else float(mf)

bestC_lr  = lin_search[lin_search.Model == 'Logistic Regression'].sort_values('PR_AUC').iloc[-1]['C']
bestC_svm = lin_search[lin_search.Model == 'SVM (linear, calibrated)'].sort_values('PR_AUC').iloc[-1]['C']
best['Logistic Regression'] = dict(C=float(bestC_lr))
best['SVM (linear, calibrated)'] = dict(C=float(bestC_svm))

def build_tuned(name):
    if name == 'XGBoost':
        return XGBClassifier(**best[name], scale_pos_weight=spw, tree_method='hist',
                             eval_metric='logloss', n_jobs=-1, random_state=RANDOM_STATE)
    if name == 'Random Forest':
        return RandomForestClassifier(**best[name], class_weight=cw, n_jobs=-1,
                                      random_state=RANDOM_STATE)
    if name == 'Logistic Regression':
        return LogisticRegression(**best[name], max_iter=2000, class_weight=cw,
                                  random_state=RANDOM_STATE)
    return CalibratedClassifierCV(LinearSVC(**best[name], class_weight=cw, max_iter=5000,
                                            random_state=RANDOM_STATE), method='sigmoid', cv=3)

tuned, tuned_val_probs, out = {}, {}, []
for name in model_names:
    m = build_tuned(name).fit(X_train, y_train)
    pv = m.predict_proba(X_val)[:, 1]
    ptr = m.predict_proba(X_train)[:, 1]
    base = results[(results.Model == name) & (results.Strategy == STRATEGY)].iloc[0]
    tuned[name], tuned_val_probs[name] = m, pv
    out.append(dict(Model=name,
                    Baseline_ROC=base['ROC_AUC'], Tuned_ROC=roc_auc_score(y_val, pv),
                    Baseline_PR=base['PR_AUC'],   Tuned_PR=average_precision_score(y_val, pv),
                    Train_ROC=roc_auc_score(y_train, ptr)))
tuned_summary = pd.DataFrame(out).round(4)
tuned_summary['Train_minus_Val_ROC'] = (tuned_summary['Train_ROC'] - tuned_summary['Tuned_ROC']).round(4)
display(tuned_summary)
print('Best parameters:'); [print(' ', k, v) for k, v in best.items()]

# ===== notebook cell 32 =====
# 6.5 Save tuning results (parameters and validation probabilities; models are refitted from the parameters)
import json
with open('best_params.json', 'w') as f:
    json.dump(best, f, indent=2)
with open('tuned_val_probs.pkl', 'wb') as f:
    pickle.dump(tuned_val_probs, f)
xgb_search.to_csv('xgb_search.csv', index=False); rf_search.to_csv('rf_search.csv', index=False)
tuned_summary.to_csv('tuned_summary.csv', index=False)
print('Saved best_params.json, tuned_val_probs.pkl and the search tables')

# ===== notebook cell 35 =====
# 7.1 Time-aware stacking
from sklearn.calibration import calibration_curve
base_names = ['Logistic Regression', 'Random Forest', 'XGBoost']
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

# X_train rows are in chronological order, so 5 position blocks = 5 consecutive time windows
blocks = np.array_split(np.arange(len(X_train)), 5)
oof_idx, oof_pred = [], []
for k in range(1, 5):                         # train on blocks < k, predict block k
    tr_idx, te_idx = np.concatenate(blocks[:k]), blocks[k]
    cols = []
    for name in base_names:
        m = build_tuned(name).fit(X_train.iloc[tr_idx], y_train[tr_idx])
        cols.append(m.predict_proba(X_train.iloc[te_idx])[:, 1])
    oof_idx.append(te_idx); oof_pred.append(np.column_stack(cols))
    print(f'window {k}/4 done')

Z_oof = np.vstack(oof_pred)
y_oof = y_train[np.concatenate(oof_idx)]
meta = LogisticRegression(C=1.0, max_iter=1000).fit(logit(Z_oof), y_oof)
print('Meta-learner weights (logit scale):', dict(zip(base_names, meta.coef_[0].round(3))),
      '| intercept', meta.intercept_.round(3))

# Base models were already refitted on the full training set in Section 6.4
Z_val = np.column_stack([tuned[n].predict_proba(X_val)[:, 1] for n in base_names])
p_stack_val = meta.predict_proba(logit(Z_val))[:, 1]

# ===== notebook cell 36 =====
# 7.2 Compare all tuned models on the validation set and apply the selection rule
cands = {n: tuned_val_probs[n] for n in model_names}
cands['Stacked ensemble'] = p_stack_val

prev = y_val.mean()
rows = []
for n, p in cands.items():
    rows.append(dict(Model=n, ROC_AUC=roc_auc_score(y_val, p), PR_AUC=average_precision_score(y_val, p),
                     Brier=brier_score_loss(y_val, p)))
comp = pd.DataFrame(rows).sort_values('PR_AUC', ascending=False).reset_index(drop=True)
comp['Brier_skill'] = 1 - comp['Brier'] / (prev * (1 - prev))   # vs always predicting the base rate
display(comp.round(4))

# Rule fixed in advance: the ensemble must beat the best single model by >= 0.005 PR-AUC to be chosen.
# Otherwise the best single model is kept (simpler, and exactly explainable with TreeSHAP).
best_single = comp[comp.Model != 'Stacked ensemble'].iloc[0]
stack_row   = comp[comp.Model == 'Stacked ensemble'].iloc[0]
gain = stack_row['PR_AUC'] - best_single['PR_AUC']
final_name = 'Stacked ensemble' if gain >= 0.005 else best_single['Model']
p_final = cands[final_name]
print(f'Ensemble PR-AUC gain over best single model ({best_single["Model"]}): {gain:+.4f}')
print('FINAL MODEL:', final_name)

# ===== notebook cell 37 =====
# 7.3 Calibration check (do predicted probabilities match observed no-show rates?)
plt.figure(figsize=(5.5, 5))
for n, p in cands.items():
    frac, mean_pred = calibration_curve(y_val, p, n_bins=10, strategy='quantile')
    plt.plot(mean_pred, frac, marker='o', label=n, lw=2.2 if n == final_name else 1)
plt.plot([0, 0.7], [0, 0.7], 'k--', label='perfect')
plt.xlabel('Mean predicted probability'); plt.ylabel('Observed no-show rate')
plt.title('Calibration on validation set'); plt.legend(fontsize=8); plt.tight_layout(); plt.show()

# ===== notebook cell 38 =====
# 7.4 Choosing the decision threshold on the validation set
prec_c, rec_c, thr_c = precision_recall_curve(y_val, p_final)
thr_f1  = best_f1_threshold(y_val, p_final)
thr_r70 = float(thr_c[np.where(rec_c[:-1] >= 0.70)[0][-1]])        # highest threshold keeping recall >= 70%
thr_top20 = float(np.sort(p_final)[::-1][int(0.20 * len(p_final))])  # flag the riskiest 20%

ops = []
for label, thr in [('F1-optimal (primary)', thr_f1), ('Recall >= 70%', thr_r70), ('Flag riskiest 20%', thr_top20)]:
    yhat = (p_final >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, yhat).ravel()
    precision = tp / (tp + fp); recall = tp / (tp + fn)
    ops.append(dict(Operating_point=label, Threshold=thr, Flagged_share=yhat.mean(),
                    Precision=precision, Recall=recall, Specificity=tn/(tn+fp),
                    F1=2*precision*recall/(precision+recall), Lift=precision/prev))
ops_table = pd.DataFrame(ops).round(4)
display(ops_table)
chosen_threshold = float(thr_f1)
print('Primary threshold fixed at', round(chosen_threshold, 4), '(chosen on validation, not on test)')

# ===== notebook cell 39 =====
# 7.5 Save the final configuration and models for the test evaluation and the SHAP / LIME sections
import joblib
final_config = dict(final_model=final_name, threshold=chosen_threshold,
                    other_thresholds=dict(recall70=thr_r70, top20=thr_top20),
                    tuned_params=best, strategy=STRATEGY)
with open('final_config.json', 'w') as f:
    json.dump(final_config, f, indent=2, default=float)
joblib.dump(dict(tuned=tuned, meta=meta, base_names=base_names), 'final_models.joblib', compress=3)
ops_table.to_csv('operating_points.csv', index=False); comp.to_csv('validation_model_comparison.csv', index=False)
print('Saved final_config.json, final_models.joblib, operating_points.csv, validation_model_comparison.csv')

# ===== notebook cell 42 =====
# 8.1 Predictions on the test set for every candidate (thresholds were fixed on validation)
from sklearn.metrics import roc_curve

def predict_any(name, X):
    if name == 'Stacked ensemble':
        Z = np.column_stack([tuned[n].predict_proba(X)[:, 1] for n in base_names])
        return meta.predict_proba(logit(Z))[:, 1]
    return tuned[name].predict_proba(X)[:, 1]

test_probs = {n: predict_any(n, X_test) for n in cands}
p_test = test_probs[final_name]
print('Final model:', final_name, '| threshold fixed on validation:', round(chosen_threshold, 4))
print('Test rows:', len(y_test), '| test no-show rate: %.4f' % y_test.mean())

# ===== notebook cell 43 =====
# 8.2 Test results: final model at all three pre-chosen operating points
rows = []
for label, thr in [('F1-optimal (primary)', chosen_threshold), ('Recall >= 70% (val)', thr_r70), ('Flag riskiest 20% (val)', thr_top20)]:
    r = evaluate(y_test, p_test, thr)
    yhat = (p_test >= thr)
    rows.append(dict(Operating_point=label, Threshold=thr, Flagged_share=yhat.mean(),
                     Precision=r['Precision'], Recall=r['Recall'], Specificity=r['Specificity'],
                     F1=r['F1'], Lift=r['Precision'] / y_test.mean()))
test_ops = pd.DataFrame(rows).round(4)
display(test_ops)

yhat = (p_test >= chosen_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, yhat).ravel()
print(f'Confusion matrix at primary threshold: TN={tn}  FP={fp}  FN={fn}  TP={tp}')

# All candidate models, each at its own validation-chosen F1 threshold
rows = []
for n in cands:
    thr = best_f1_threshold(y_val, cands[n])
    rows.append(dict(Model=n, **evaluate(y_test, test_probs[n], thr)))
test_all = pd.DataFrame(rows).sort_values('PR_AUC', ascending=False).round(4).reset_index(drop=True)
print('\nAll candidate models on the test set (reporting only; selection was made on validation)')
display(test_all)

# ===== notebook cell 44 =====
# 8.3 Stability: validation vs test for the final model
rv = evaluate(y_val, p_final, chosen_threshold); rt = evaluate(y_test, p_test, chosen_threshold)
stab = pd.DataFrame({'Validation': rv, 'Test': rt}).drop(index='Threshold')
stab['Change'] = stab['Test'] - stab['Validation']
display(stab.round(4))

# Day-by-day stability inside the test window
test_days = test['AppointmentDay'].dt.date.values if 'test' in globals() else None
if test_days is not None:
    for d in np.unique(test_days):
        m = test_days == d
        # ROC-AUC needs both classes; skip it (as the bootstrap does) if a day has only one
        auc_txt = ('%.4f' % roc_auc_score(y_test[m], p_test[m])
                   if y_test[m].min() != y_test[m].max() else 'n/a (only one class on this day)')
        print(d, '| rows', m.sum(), '| no-show rate %.3f' % y_test[m].mean(), '| ROC-AUC', auc_txt)

# ===== notebook cell 45 =====
# 8.4 Bootstrap 95% confidence intervals (1,000 resamples of the test set)
def boot_ci(y, p, thr, B=1000, seed=RANDOM_STATE):
    rs = np.random.default_rng(seed); n = len(y); out = []
    for _ in range(B):
        idx = rs.integers(0, n, n); yb, pb = y[idx], p[idx]
        if yb.min() == yb.max(): continue
        h = pb >= thr
        tp_ = (h & (yb == 1)).sum(); fp_ = (h & (yb == 0)).sum()
        fn_ = (~h & (yb == 1)).sum(); tn_ = (~h & (yb == 0)).sum()
        out.append([roc_auc_score(yb, pb), average_precision_score(yb, pb),
                    tp_ / max(tp_ + fp_, 1), tp_ / max(tp_ + fn_, 1), tn_ / max(tn_ + fp_, 1)])
    arr = np.array(out); lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
    return pd.DataFrame({'Metric': ['ROC_AUC','PR_AUC','Precision','Recall','Specificity'],
                         'Estimate': [rt['ROC_AUC'], rt['PR_AUC'], rt['Precision'], rt['Recall'], rt['Specificity']],
                         'CI_low': lo, 'CI_high': hi})
ci_table = boot_ci(y_test, p_test, chosen_threshold)
display(ci_table.round(4))

# ===== notebook cell 46 =====
# 8.5 Curves and calibration on the test set (raw vs Platt-recalibrated on validation)
platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_final).reshape(-1, 1), y_val)
p_test_cal = platt.predict_proba(logit(p_test).reshape(-1, 1))[:, 1]
print('Test Brier  raw: %.4f | recalibrated on validation: %.4f' %
      (brier_score_loss(y_test, p_test), brier_score_loss(y_test, p_test_cal)))
print('Mean predicted risk  raw: %.4f | recalibrated: %.4f | observed no-show rate: %.4f' %
      (p_test.mean(), p_test_cal.mean(), y_test.mean()))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for n, p in test_probs.items():
    fpr, tpr, _ = roc_curve(y_test, p); ax[0].plot(fpr, tpr, lw=2.2 if n == final_name else 1, label=n)
    pr, rc, _ = precision_recall_curve(y_test, p); ax[1].plot(rc, pr, lw=2.2 if n == final_name else 1, label=n)
ax[0].plot([0,1],[0,1],'k--'); ax[0].set(title='ROC curves (test)', xlabel='False positive rate', ylabel='True positive rate')
ax[1].axhline(y_test.mean(), color='k', ls='--'); ax[1].set(title='Precision-recall curves (test)', xlabel='Recall', ylabel='Precision')
ax[0].legend(fontsize=7)
for p, lab in [(p_test, 'raw'), (p_test_cal, 'recalibrated')]:
    f, m = calibration_curve(y_test, p, n_bins=10, strategy='quantile'); ax[2].plot(m, f, marker='o', label=lab)
ax[2].plot([0,0.6],[0,0.6],'k--'); ax[2].set(title='Calibration of ' + final_name + ' (test)', xlabel='Mean predicted', ylabel='Observed'); ax[2].legend()
plt.tight_layout(); plt.show()

# ===== notebook cell 47 =====
# 8.6 Save test results for the SHAP / LIME sections and for Chapter 4
test_ops.to_csv('test_operating_points.csv', index=False); test_all.to_csv('test_all_models.csv', index=False)
ci_table.to_csv('test_bootstrap_ci.csv', index=False); stab.to_csv('validation_vs_test.csv')
with open('test_probs.pkl', 'wb') as f:
    pickle.dump(dict(test_probs=test_probs, p_test_cal=p_test_cal, y_test=y_test), f)
print('Saved test result tables and test_probs.pkl')

# ===== notebook cell 50 =====
# 9.0 Setup
import shap, time
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

explain_name = final_name if final_name in ('XGBoost', 'Random Forest') else 'XGBoost'
model_x = tuned[explain_name]
print('Selected final model (Section 7.2):', final_name)
print('Model explained by SHAP and LIME:   ', explain_name)
if explain_name == final_name:
    print('The explained model IS the selected final model, so the explanations describe the model that was evaluated on the test set.')
else:
    # Only happens if the stacked ensemble wins the selection rule. State this in the write-up and word RQ2 accordingly.
    print('WARNING: the stacked ensemble was selected but is not TreeSHAP-compatible, so its strongest base model (XGBoost) is explained instead.')
    print('         Report this in Chapters 4 and 5 and word RQ2 as "...explaining the predictions of the XGBoost base model".')

# SHAP: exact TreeSHAP over the whole test set
explainer = shap.TreeExplainer(model_x)
t0 = time.time()
sv = explainer(X_test)
if sv.values.ndim == 3:            # Random Forest returns one slice per class
    sv = sv[..., 1]
shap_time_per_case = (time.time() - t0) / len(X_test)
print('SHAP computed for %d test cases (%.5f s per case)' % (len(X_test), shap_time_per_case))

# ===== notebook cell 51 =====
# 9.1 Global SHAP explanation
shap.plots.bar(sv, max_display=15)
shap.plots.beeswarm(sv, max_display=15)
global_shap = pd.Series(np.abs(sv.values).mean(0), index=feature_names).sort_values(ascending=False)
display(global_shap.head(15).round(4).to_frame('mean_abs_SHAP'))

# ===== notebook cell 52 =====
# 9.2 Dependence plots for the strongest drivers (values are standardised)
for f in global_shap.index[:3]:
    shap.plots.scatter(sv[:, f], color=sv)

# ===== notebook cell 53 =====
# 9.3 LIME on a random sample of 300 test cases (same cases as SHAP)
rs = np.random.default_rng(RANDOM_STATE)
sample_idx = rs.choice(len(X_test), 300, replace=False)
Xs = X_test.iloc[sample_idx]
shap_s = sv.values[sample_idx]

cat_idx = [i for i, c in enumerate(feature_names) if c not in numeric_cols]
lime_background = X_train.sample(20000, random_state=RANDOM_STATE).values
predict_fn = lambda a: model_x.predict_proba(pd.DataFrame(a, columns=feature_names))

def make_lime(seed):
    return LimeTabularExplainer(lime_background, feature_names=feature_names,
                                class_names=['Attend', 'No-show'], categorical_features=cat_idx,
                                mode='classification', discretize_continuous=True, random_state=seed)

def lime_weights(exp, x, num_samples=3000):
    e = exp.explain_instance(x, predict_fn, num_features=len(feature_names), num_samples=num_samples)
    w = np.zeros(len(feature_names))
    for i, wt in e.as_map()[1]:
        w[i] = wt
    return w

lime_main = make_lime(RANDOM_STATE)
t0 = time.time()
lime_s = np.array([lime_weights(lime_main, x) for x in Xs.values])
lime_time_per_case = (time.time() - t0) / len(Xs)
print('LIME computed for %d cases (%.3f s per case)' % (len(Xs), lime_time_per_case))

# ===== notebook cell 54 =====
# 9.4 Local agreement between SHAP and LIME (per case, then averaged)
def topk(a, k): return set(np.argsort(-np.abs(a))[:k])
rows = []
for s, l in zip(shap_s, lime_s):
    t5s, t5l = topk(s, 5), topk(l, 5)
    common = list(t5s & t5l)
    rows.append(dict(
        rank_corr_abs=spearmanr(np.abs(s), np.abs(l)).correlation,                  # same importance ordering?
        top1_same=float(np.argmax(np.abs(s)) == np.argmax(np.abs(l))),
        top5_overlap=len(common) / 5,
        sign_agree_top5=np.mean(np.sign(s[common]) == np.sign(l[common])) if common else np.nan,
        share_corr=np.corrcoef(s / np.abs(s).sum(), l / np.abs(l).sum())[0, 1]))   # direction + relative size
agree = pd.DataFrame(rows)
display(agree.describe().loc[['mean', '50%', 'min', 'max']].round(3))

# ===== notebook cell 55 =====
# 9.5 Global usefulness: do the two methods rank features the same way overall?
g = pd.DataFrame({'feature': feature_names,
                  'mean_abs_SHAP': np.abs(shap_s).mean(0), 'mean_abs_LIME': np.abs(lime_s).mean(0)})
g['SHAP_share'] = g['mean_abs_SHAP'] / g['mean_abs_SHAP'].sum()
g['LIME_share'] = g['mean_abs_LIME'] / g['mean_abs_LIME'].sum()
g = g.sort_values('SHAP_share', ascending=False).reset_index(drop=True)
display(g.head(10).round(4))
global_rank_corr = spearmanr(g['mean_abs_SHAP'], g['mean_abs_LIME']).correlation
print('Spearman correlation of global feature rankings (SHAP vs LIME): %.3f' % global_rank_corr)

top10 = g.head(10).iloc[::-1]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
ax[0].barh(top10['feature'], top10['SHAP_share']); ax[0].set_title('Global importance share: SHAP')
ax[1].barh(top10['feature'], top10['LIME_share'], color='tab:orange'); ax[1].set_title('Global importance share: LIME')
plt.tight_layout(); plt.show()

# ===== notebook cell 56 =====
# 9.6 Stability: repeat LIME with 3 different random seeds on 30 cases (TreeSHAP is deterministic)
n_stab = 30
runs = [np.array([lime_weights(make_lime(sd), x) for x in Xs.values[:n_stab]]) for sd in (1, 2, 3)]
pair_rho, pair_t5 = [], []
for a, b in [(0, 1), (0, 2), (1, 2)]:
    for i in range(n_stab):
        pair_rho.append(spearmanr(np.abs(runs[a][i]), np.abs(runs[b][i])).correlation)
        pair_t5.append(len(topk(runs[a][i], 5) & topk(runs[b][i], 5)) / 5)
shap_again = explainer(Xs.iloc[:n_stab])
shap_again_vals = shap_again.values[..., 1] if shap_again.values.ndim == 3 else shap_again.values
shap_identical = np.allclose(shap_again_vals, shap_s[:n_stab])
print('LIME across seeds: mean rank corr %.3f | mean top-5 overlap %.3f' % (np.nanmean(pair_rho), np.mean(pair_t5)))
print('SHAP identical on repeat run:', shap_identical)

# ===== notebook cell 57 =====
# 9.7 Four illustrative test cases (one per confusion-matrix outcome)
yhat_t = (p_test >= chosen_threshold).astype(int)
groups = {'True positive': (yhat_t == 1) & (y_test == 1), 'False positive': (yhat_t == 1) & (y_test == 0),
          'False negative': (yhat_t == 0) & (y_test == 1), 'True negative': (yhat_t == 0) & (y_test == 0)}
show_cols = [c for c in ['LeadTime', 'Age', 'PrevNoShows', 'PrevNoShowRate', 'SMSReceived', 'ComorbidityCount'] if 'test_m' in globals()]
for label, mask in groups.items():
    i = int(np.random.default_rng(7).choice(np.where(mask)[0]))
    print('\n=====', label, '| test row', i, '| model P(no-show) = %.3f' % p_test[i], '| actual no-show =', y_test[i])
    if show_cols:
        print(test_m.iloc[i][show_cols].to_dict())
    plt.close('all')
    try:
        shap.plots.waterfall(sv[i], max_display=10)
    except Exception as err:               # fall back to a simple bar chart if a plot version clashes
        print('(waterfall unavailable:', repr(err)[:80], ') showing bar chart instead')
        plt.close('all'); shap.plots.bar(sv[i], max_display=10)
    e = lime_main.explain_instance(X_test.values[i], predict_fn, num_features=10, num_samples=3000)
    e.as_pyplot_figure(label=1); plt.tight_layout(); plt.show()

# ===== notebook cell 58 =====
# 9.8 One-table summary of the SHAP vs LIME comparison (for Chapter 4)
m = agree.mean(numeric_only=True)
comparison = pd.DataFrame([
    ['Feature attribution: same importance ordering (mean rank corr)', round(m['rank_corr_abs'], 3), '', ''],
    ['Feature attribution: same #1 feature', round(m['top1_same'], 3), '', ''],
    ['Feature attribution: top-5 overlap', round(m['top5_overlap'], 3), '', ''],
    ['Direction: sign agreement on shared top-5 features', round(m['sign_agree_top5'], 3), '', ''],
    ['Magnitude: correlation of normalised attributions', round(m['share_corr'], 3), '', ''],
    ['Global usefulness: rank corr of global importance', round(global_rank_corr, 3), '', ''],
    ['Stability across repeated runs', 'SHAP: identical' if shap_identical else 'SHAP: varies',
        'LIME rank corr %.3f' % np.nanmean(pair_rho), 'LIME top-5 overlap %.3f' % np.mean(pair_t5)],
    ['Compute time per case (s)', round(shap_time_per_case, 5), round(lime_time_per_case, 3), ''],
], columns=['Criterion', 'Value / SHAP', 'LIME', 'Extra'])
display(comparison)

comparison.to_csv('shap_lime_comparison.csv', index=False); agree.to_csv('shap_lime_per_case_agreement.csv', index=False)
g.to_csv('global_importance_shap_lime.csv', index=False)
print('Saved shap_lime_comparison.csv, shap_lime_per_case_agreement.csv, global_importance_shap_lime.csv')

# ===== notebook cell 59 =====
# 9.9 Sensitivity check: does LIME's picture change with a different discretisation of numeric features?
bg = X_train.sample(20000, random_state=RANDOM_STATE)
bg_labels = pd.Series(y_train, index=X_train.index).loc[bg.index].values

def agreement_means(Sh, W):
    out = []
    for s, l in zip(Sh, W):
        t5s, t5l = topk(s, 5), topk(l, 5); common = list(t5s & t5l)
        out.append([spearmanr(np.abs(s), np.abs(l)).correlation, float(np.argmax(np.abs(s)) == np.argmax(np.abs(l))),
                    len(common) / 5, np.mean(np.sign(s[common]) == np.sign(l[common])) if common else np.nan,
                    np.corrcoef(s / np.abs(s).sum(), l / np.abs(l).sum())[0, 1]])
    return np.nanmean(np.array(out), axis=0)

variants = {'quartile (used above)': lime_s}
for disc in ['decile', 'entropy']:
    kw = dict(training_labels=bg_labels) if disc == 'entropy' else {}
    ex = LimeTabularExplainer(bg.values, feature_names=feature_names, class_names=['Attend', 'No-show'],
                              categorical_features=cat_idx, mode='classification', discretize_continuous=True,
                              discretizer=disc, random_state=RANDOM_STATE, **kw)
    variants[disc] = np.array([lime_weights(ex, x) for x in Xs.values])

li = feature_names.index('LeadTime')
rows = []
for name, W in variants.items():
    a = agreement_means(shap_s, W)
    share = np.abs(W).mean(0); share = share / share.sum()
    rows.append(dict(LIME_discretiser=name, rank_corr=a[0], same_top1=a[1], top5_overlap=a[2], sign_agree=a[3],
                     share_corr=a[4], LeadTime_share=share[li],
                     global_rank_corr=spearmanr(np.abs(shap_s).mean(0), np.abs(W).mean(0)).correlation))
sens = pd.DataFrame(rows).round(3)
print('SHAP gives LeadTime a global share of %.3f' % (np.abs(shap_s).mean(0)[li] / np.abs(shap_s).mean(0).sum()))
display(sens)
sens.to_csv('lime_sensitivity.csv', index=False)

# ===== notebook cell 60 =====
# 9.10 Local fidelity and explanation simplicity (how the "transparency / interpretability" criterion is measured)
#
# (a) LOCAL FIDELITY. LIME explains a case by fitting a small weighted linear model around it. LIME's own score for that
#     model is a weighted R^2: 1.0 = the simple model reproduces the XGBoost probabilities around the case perfectly,
#     0 = no better than a constant. TreeSHAP needs no such check because its values add up exactly to the model output
#     (verified below).
# (b) EXPLANATION SIMPLICITY. How many features are needed to account for at least 80% of the total absolute attribution
#     of a case? Fewer features = a shorter, easier explanation to read.
# Both are computed on the same 300 test cases used in 9.3-9.9.

fresh = make_lime(RANDOM_STATE)           # same seed and order as 9.3, so these are the same 300 explanations as lime_s
r2_all, w_all = [], []
for x in Xs.values:
    e = fresh.explain_instance(x, predict_fn, num_features=len(feature_names), num_samples=3000)
    r2_all.append(e.score)
    w = np.zeros(len(feature_names))
    for i, wt in e.as_map()[1]: w[i] = wt
    w_all.append(w)
print('Same explanations as 9.3 (all LIME weights identical):', bool(np.allclose(np.array(w_all), lime_s)))

shown = make_lime(RANDOM_STATE + 1)       # the 10-feature explanation a reader actually sees in the charts
r2_top10 = [shown.explain_instance(x, predict_fn, num_features=10, num_samples=3000).score for x in Xs.values]

def describe(a):
    a = np.asarray(a, dtype=float)
    return dict(mean=a.mean(), median=np.median(a), p25=np.percentile(a, 25), p75=np.percentile(a, 75), min=a.min(), max=a.max())

fidelity = pd.DataFrame({'LIME, all 33 features': describe(r2_all), 'LIME, top 10 features (as plotted)': describe(r2_top10)}).T
print('\nLocal fidelity of LIME: weighted R^2 of the local surrogate (300 test cases)')
display(fidelity.round(3))
print('Share of cases with R^2 >= 0.5: all features %.1f%% | top 10 %.1f%%' % (100 * np.mean(np.array(r2_all) >= 0.5), 100 * np.mean(np.array(r2_top10) >= 0.5)))

# SHAP is exact: base value + sum of SHAP values must equal the model's raw (log-odds) output
try:
    margin = model_x.predict(X_test, output_margin=True)
    add_err = np.abs(sv.base_values + sv.values.sum(1) - margin).max()
    print('SHAP additivity: largest |base + sum(SHAP) - model output| over all %d test cases = %.2e (log-odds)' % (len(X_test), add_err))
except Exception as err:
    print('(additivity check skipped:', repr(err)[:80], ')')

def n_for_share(a, share=0.80):
    s = np.sort(np.abs(a))[::-1]
    return int(np.searchsorted(np.cumsum(s) / s.sum(), share) + 1)

n_shap = np.array([n_for_share(a) for a in shap_s]); n_lime = np.array([n_for_share(a) for a in lime_s])
simp = pd.DataFrame({'SHAP': describe(n_shap), 'LIME': describe(n_lime)}).T
print('\nExplanation simplicity: number of features needed to account for >= 80%% of absolute attribution (of %d)' % len(feature_names))
display(simp.round(2))
print('SHAP needs fewer features than LIME in %.1f%% of cases, the same number in %.1f%%, more in %.1f%%' %
      (100 * np.mean(n_shap < n_lime), 100 * np.mean(n_shap == n_lime), 100 * np.mean(n_shap > n_lime)))

# ===== notebook cell 64 =====
# 10.1 Export the final model for the web app (exact tree structure, no retraining needed)
import json
if final_name != 'XGBoost':
    raise RuntimeError('The web app evaluates XGBoost trees; the selected final model is ' + final_name + '. Recompute the thresholds and Platt scaling for XGBoost before exporting.')
bst = tuned['XGBoost'].get_booster(); bst.save_model('_m.json')
J = json.load(open('_m.json'))['learner']
trees = [dict(f=[-1 if l < 0 else int(s) for l, s in zip(t['left_children'], t['split_indices'])],
              t=[float(x) for x in t['split_conditions']], l=[int(x) for x in t['left_children']],
              r=[int(x) for x in t['right_children']],
              v=[float(x) if l < 0 else 0.0 for l, x in zip(t['left_children'], t['split_conditions'])],
              c=[float(x) for x in t['sum_hessian']])
         for t in J['gradient_booster']['model']['trees']]
b = float(str(J['learner_model_param']['base_score']).strip('[]'))
nt = preprocess.named_transformers_
raw_cols = ['Age','Gender','Neighborhood','Scholarship','Hypertension','Diabetes','Alcoholism','Handicap',
            'SMSReceived','LeadTime','ApptDayOfWeek','SchedDayOfWeek','PrevAppts','PrevNoShows']
def to_inp(r):
    return dict(age=int(r.Age), male=int(r.Gender == 'M'), nbh=r.Neighborhood, sch=int(r.Scholarship), hyp=int(r.Hypertension),
                dia=int(r.Diabetes), alc=int(r.Alcoholism), hand=int(r.Handicap), sms=int(r.SMSReceived), lead=int(r.LeadTime),
                apptDow=r.ApptDayOfWeek, schedDow=r.SchedDayOfWeek, prevAppts=int(r.PrevAppts), prevNoShows=int(r.PrevNoShows))
chk = test_m.iloc[:25]
pc = tuned['XGBoost'].predict_proba(X_test.iloc[:25])[:, 1]
export = dict(trees=trees, base=float(np.log(b / (1 - b))), names=list(feature_names),
              num=dict(cols=list(numeric_cols), mean=nt['num'].mean_.tolist(), scale=nt['num'].scale_.tolist()),
              nb={k: float(v) for k, v in freq_map.items()},
              platt=[float(platt.coef_[0][0]), float(platt.intercept_[0])],
              thr=dict(f1=float(chosen_threshold), r70=float(thr_r70), top20=float(thr_top20)),
              meta=dict(roc=float(rt['ROC_AUC']), pr=float(rt['PR_AUC'])),
              checks=[dict(x=to_inp(r), p=float(p)) for (_, r), p in zip(chk[raw_cols + ['Y']].iterrows(), pc)])
json.dump(export, open('noshow_model.json', 'w'))
print('Exported', len(trees), 'trees; size', round(len(json.dumps(export)) / 1024), 'KB')

try:
    from google.colab import files; files.download('noshow_model.json')
except ImportError:
    print('Not running in Colab: noshow_model.json is in the working folder.')

