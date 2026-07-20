"""
train.py
========
BM 264 Makine Öğrenmesi Projesi — Model Eğitim Betiği

Bu betik:
  1) X_train.csv ve y_train.csv'yi okur
  2) preprocess.preprocess() ile ön işleme uygular (sütun isimleri korunur)
  3) Bir Pipeline eğitir: x4'ü düşür → polinom 3 genişletme → RandomForest
  4) 5-fold CV ile composite (0.5*Acc + 0.5*MacroF1) skor bildirir
  5) Pipeline'ı model.joblib ve model_metadata.json olarak kaydeder

Tasarım kararları
-----------------
• Veri keşfi (özet):
    - 2250 örnek, 4 ham öznitelik (x1, x2, x3, x4_irrelevant), 3 sınıf.
    - Sınıflar 60/30/10 oranında (orta düzeyde dengesiz).
    - NaN/duplicate/inf yok.
    - Ham öznitelikler çok farklı ölçeklerde (x2/x3 ~3 mertebe büyük).
    - x4_irrelevant: sınıflarla MI = 0.0000, koşullu dağılım birebir aynı —
      gürültü öznitelik.
    - Tüm tek değişkenli Pearson korelasyonları ~0 → sınıflar lineer
      ayrılabilir DEĞİL.
    - Kuadratik/kübik çapraz terimler (x_i*x_j vb.) güçlü sinyal taşıyor;
      bu da problem tanımındaki "3 boyutlu geometrik yapı" ipucuyla
      tutarlı (3B kuadrik/kübik yüzeylerle ayrılmış bölgeler).

• Ön işleme (preprocess.py içinde):
    - x1, x2, x3 z-skor ile standartlaştırılır (eğitim MEAN/STD sabit kodlu).
    - x4_irrelevant olduğu gibi geçer; çıktı sütun isimleri ve sırası
      X_test ile birebir aynı kalır (şartname gereği).

• Model (sklearn.Pipeline):
    - Adım A: ColumnDropper("x4_irrelevant")  → 3 sütun kalır
    - Adım B: PolynomialFeatureExpander       → 19 öznitelik
    - Adım C: RandomForestClassifier(n_estimators=300)
    Her iki custom sınıf da preprocess.py içinde tanımlı. joblib bu
    modülden import edildikleri için pickle bunları "preprocess.X" yolunda
    kaydeder; değerlendirici joblib.load çağırdığında Python otomatik
    `import preprocess` yapar ve sınıfları bulur.

• Performans (5-fold stratified CV):
    composite ≈ 0.978 ± 0.013.
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_val_predict)
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             classification_report, confusion_matrix)

# CRITICAL: Custom Pipeline sınıflarını preprocess modülünden import et.
# Böylece joblib pickle bunları "preprocess.ColumnDropper" diye kaydeder
# ve değerlendirici joblib.load() çağırdığında Python `preprocess`
# modülünü otomatik import ederek sınıfları bulur.
from preprocess import (preprocess,
                        ColumnDropper,
                        PolynomialFeatureExpander)


# =============================================================================
# 1) Veri yükle
# =============================================================================
print("Eğitim verisi okunuyor...")
X_raw = pd.read_csv("X_train.csv")
y     = pd.read_csv("y_train.csv")["label"].values
print(f"  X_train şekli       : {X_raw.shape}")
print(f"  Sınıf dağılımı (0,1,2): {np.bincount(y).tolist()}")


# =============================================================================
# 2) Ön işleme uygula (preprocess.py ile aynı dönüşüm)
# =============================================================================
print("\nÖn işleme uygulanıyor (preprocess.preprocess)...")
X_p = preprocess(X_raw)
print(f"  Ön işlenmiş şekil   : {X_p.shape}")
print(f"  Sütunlar (X_test ile aynı): {list(X_p.columns)}")


# =============================================================================
# 3) Pipeline: ColumnDropper → PolynomialFeatureExpander → RandomForest
# =============================================================================
expected_cols = list(X_p.columns)  # ["x1", "x2", "x3", "x4_irrelevant"]

model = Pipeline([
    ("drop_irrelevant", ColumnDropper(
        columns_to_drop=["x4_irrelevant"],
        expected_input_cols=expected_cols,
    )),
    ("poly_expand", PolynomialFeatureExpander()),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )),
])


# =============================================================================
# 4) 5-fold Stratified CV
# =============================================================================
def composite_score(y_true, y_pred):
    return 0.5*accuracy_score(y_true, y_pred) + \
           0.5*f1_score(y_true, y_pred, average="macro")

scorer = make_scorer(composite_score)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== 5-fold Stratified CV ===")
# Pipeline'a DataFrame veriyoruz; ColumnDropper DataFrame'i de tanıyor.
sc = cross_val_score(model, X_p, y, cv=cv, scoring=scorer, n_jobs=-1)
print(f"Composite skor: {sc.mean():.4f} ± {sc.std():.4f}")
print(f"Fold skorları : {[f'{s:.4f}' for s in sc]}")

yp_cv = cross_val_predict(model, X_p, y, cv=cv, n_jobs=-1)
print("\nClassification report (CV tahminleri):")
print(classification_report(y, yp_cv, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y, yp_cv))


# =============================================================================
# 5) Tüm eğitim setiyle eğit ve kaydet
# =============================================================================
print("\nTüm eğitim verisiyle Pipeline yeniden eğitiliyor...")
model.fit(X_p, y)

joblib.dump(model, "model.joblib")
print("  model.joblib kaydedildi.")

with open("model_metadata.json", "w") as f:
    json.dump({"framework": "sklearn", "model_file": "model.joblib"}, f, indent=2)
print("  model_metadata.json kaydedildi.")
