import os, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

train = np.load(ROOT / 'processed' / 'corrected_clean' / 'train.npz')
test = np.load(ROOT / 'processed' / 'corrected_clean' / 'test.npz')
X_tr, y_tr = train['X'], train['y']
X_te, y_te = test['X'], test['y']

baselines = {}

# 1. Majority class baseline
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_tr, y_tr)
preds_dummy = dummy.predict(X_te)
baselines['Majority Class (Zero-Rule)'] = {
    'accuracy': float(accuracy_score(y_te, preds_dummy)),
    'macro_f1': float(f1_score(y_te, preds_dummy, average='macro', zero_division=0)),
    'balanced_acc': float(balanced_accuracy_score(y_te, preds_dummy)),
    'cm': confusion_matrix(y_te, preds_dummy).tolist()
}

# 2. Logistic Regression baseline
lr = LogisticRegression(max_iter=500, random_state=42, C=1.0)
lr.fit(X_tr, y_tr)
preds_lr = lr.predict(X_te)
baselines['Logistic Regression (L2)'] = {
    'accuracy': float(accuracy_score(y_te, preds_lr)),
    'macro_f1': float(f1_score(y_te, preds_lr, average='macro', zero_division=0)),
    'balanced_acc': float(balanced_accuracy_score(y_te, preds_lr)),
    'cm': confusion_matrix(y_te, preds_lr).tolist()
}

# 3. Random Forest baseline
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
preds_rf = rf.predict(X_te)
baselines['Random Forest (100 Trees)'] = {
    'accuracy': float(accuracy_score(y_te, preds_rf)),
    'macro_f1': float(f1_score(y_te, preds_rf, average='macro', zero_division=0)),
    'balanced_acc': float(balanced_accuracy_score(y_te, preds_rf)),
    'cm': confusion_matrix(y_te, preds_rf).tolist()
}

for name, res in baselines.items():
    acc = res['accuracy']
    f1 = res['macro_f1']
    bal = res['balanced_acc']
    print(f'{name} -> Acc: {acc:.4f}, Macro-F1: {f1:.4f}, Bal-Acc: {bal:.4f}')

with open(str(ROOT / 'results' / 'classical_baselines_benchmark.json'), 'w') as f:
    json.dump(baselines, f, indent=2)
print('Baselines saved to results/classical_baselines_benchmark.json')
