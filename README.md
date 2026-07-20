# geometric-3class

BM 264 Makine Öğrenmesi dersi kapsamında hazırlanmış, 3 sınıflı geometrik veri seti üzerinde çalışan bir sınıflandırma projesi.

## Veri seti

- 2250 örnek, 4 ham öznitelik (`x1`, `x2`, `x3`, `x4_irrelevant`), 3 sınıf.
- Sınıflar dengesiz dağılıyor (~60/30/10).
- `x4_irrelevant` sınıflarla bilgi taşımıyor (mutual information ≈ 0) ve modelde kullanılmıyor.
- Öznitelikler arasındaki tekil (lineer) korelasyonlar ~0; ayrım kuadratik/kübik çapraz terimlerle sağlanıyor — 3 boyutlu geometrik yapılarla ayrılmış sınıflara işaret ediyor.

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `preprocess.py` | `x1, x2, x3`'ü eğitim setinden sabitlenmiş ortalama/std ile standartlaştırır; `x4_irrelevant` sütununu değiştirmeden korur (çıktı formatı `X_test.csv` ile birebir aynı kalmalı). |
| `train.py` | Ön işlenmiş veriyi okuyup `ColumnDropper → PolynomialFeatureExpander → RandomForestClassifier` pipeline'ını eğitir, 5-fold stratified CV ile değerlendirir, `model.joblib` ve `model_metadata.json` olarak kaydeder. |
| `model.joblib` | Eğitilmiş sklearn pipeline'ı. |
| `model_metadata.json` | Model dosyasına ait basit meta veri (`framework`, `model_file`). |

## Model

Pipeline üç adımdan oluşuyor:

1. **ColumnDropper** — `x4_irrelevant` sütununu atar.
2. **PolynomialFeatureExpander** — kalan 3 özniteliği 19 özniteliğe genişletir (kuadratik/kübik terimler).
3. **RandomForestClassifier** — 300 ağaç.

5-fold stratified CV'de composite skor (0.5×Accuracy + 0.5×Macro-F1) ≈ **0.978 ± 0.013**.

## Çalıştırma

```bash
python preprocess.py   # X_test.csv -> X_test_preprocessed.csv
python train.py         # X_train.csv, y_train.csv -> model.joblib, model_metadata.json
```

`model.joblib`, `preprocess.py` içindeki `ColumnDropper` ve `PolynomialFeatureExpander` sınıflarına bağımlı olduğundan, modeli `joblib.load` ile yüklerken bu dosyanın aynı dizinde bulunması gerekir.
