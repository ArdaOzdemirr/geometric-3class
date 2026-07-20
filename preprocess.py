"""
preprocess.py
=============
Makine Öğrenmesi Ödevi için ön işleme hattı şablonu.

Talimatlar
----------
1. PARAMETRELER bölümünde eğitim setinizden hesapladığınız değerleri girin.
2. `preprocess()` fonksiyonunda ön işleme adımlarınızı uygulayın.
3. Giriş/çıkış dosya adlarını veya genel yapıyı değiştirmeyin.
4. Bu kodu gönderim klasörünüzden çalıştırın:

       python preprocess.py

   Aynı dizindeki X_test.csv dosyasını okuyarak X_test_preprocessed.csv
   dosyasını aynı dizine kaydeder.

Notlar
------
- Eğitim setinde hesapladığınız parametreler
  buraya sabit kodlanmalıdır — harici dosya yüklemeyin.
- Çıktı, X_test.csv ile aynı satır sayısına sahip
  olmalıdır.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# =============================================================================
# PARAMETRELER — eğitim setinde hesapladığınız değerleri buraya girin
# =============================================================================
#
# Uyguladığımız ön işleme:
#   • x1, x2, x3 sütunları, eğitim setinden hesaplanan ortalama (MEAN) ve
#     standart sapma (STD, ddof=0) ile z-skor olarak standartlaştırılır.
#     Ham veride x1 ∈ [-1.6, 1.7], x2 ∈ [-1.6k, 1.7k], x3 ∈ [-150, 5100];
#     ölçekler çok farklı, eşit ağırlık için standardize gerekir.
#   • x4_irrelevant: eğitim verisinde sınıflarla mutual information ~ 0,
#     koşullu dağılım birebir aynı — bilgi taşımıyor. Şartname "çıktının
#     X_test ile aynı sütun isimleri/sırasına sahip olması" gerektiğini
#     söylediği için ön işlenmiş çıktıda KORUNUR (içeriği olduğu gibi
#     geçirilir). Model dosyasındaki Pipeline ilk adım olarak bu sütunu
#     atar — yani çıkarımı etkilemez ama gönderim formatına uyar.
#
# Aşağıdaki MEAN/STD değerleri X_train.csv üzerinden hesaplandı
# (pandas .mean() ve .std(ddof=0)).

FEATURE_COLS = ["x1", "x2", "x3"]

# Eğitim setinden hesaplanan ortalamalar
MEAN = np.array([
    -0.006506937403309212,   # x1
    18.571989017787065,      # x2
    2501.1389032881684,      # x3
])

# Eğitim setinden hesaplanan standart sapmalar (ddof=0, populasyon)
STD = np.array([
    0.7460677471848565,      # x1
    747.7202344503135,       # x2
    1451.4157734533621,      # x3
])

# Modelin Pipeline'ında inference sırasında atılacak sütun(lar)
DROP_COLS = ["x4_irrelevant"]


# =============================================================================
# Modelin Pipeline'ı tarafından kullanılan dönüşüm sınıfları
# -----------------------------------------------------------------------------
# Bu sınıflar burada preprocess.py içinde tanımlıdır çünkü joblib (pickle)
# modeli yüklerken sınıf tanımlarına erişebilmek zorundadır. preprocess.py
# zaten gönderim klasörünün parçası olduğundan, değerlendirici joblib.load
# çağırdığında Python otomatik olarak bu modülü import eder ve sınıfları
# bulur.
# =============================================================================

class ColumnDropper(BaseEstimator, TransformerMixin):
    """Listelenen sütun isimlerini (varsa) düşürür. DataFrame veya 2D numpy
    girdiyle çalışır. numpy halinde, `expected_input_cols` sırasındaki
    konumlara göre düşürür."""

    def __init__(self, columns_to_drop=None, expected_input_cols=None):
        # sklearn clone uyumluluğu: parametreleri olduğu gibi sakla.
        self.columns_to_drop = columns_to_drop
        self.expected_input_cols = expected_input_cols

    def _resolved(self):
        cd = self.columns_to_drop if self.columns_to_drop is not None \
             else list(DROP_COLS)
        ec = self.expected_input_cols if self.expected_input_cols is not None \
             else ["x1", "x2", "x3", "x4_irrelevant"]
        return list(cd), list(ec)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cd, ec = self._resolved()
        if isinstance(X, pd.DataFrame):
            keep = [c for c in X.columns if c not in cd]
            return X[keep].values.astype(float)
        X = np.asarray(X, dtype=float)
        keep_idx = [i for i, c in enumerate(ec) if c not in cd]
        return X[:, keep_idx]


class PolynomialFeatureExpander(BaseEstimator, TransformerMixin):
    """3 sütunlu girişten (z1, z2, z3) 19 polinom öznitelik üretir:
    orijinaller + kareler + küpler + ikili ve üçlü çarpımlar.
    Ham veride tek değişkenli korelasyonlar ~0 (sınıflar lineer
    ayrılabilir değil); kuadratik/kübik çapraz terimler asıl sinyali
    içerir."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != 3:
            raise ValueError(
                f"PolynomialFeatureExpander 3 sütun bekliyor, "
                f"alınan: {X.shape[1]}"
            )
        z1, z2, z3 = X[:, 0], X[:, 1], X[:, 2]
        return np.column_stack([
            z1, z2, z3,
            z1**2, z2**2, z3**2,
            z1*z2, z1*z3, z2*z3,
            z1**3, z2**3, z3**3,
            z1**2 * z2, z1**2 * z3,
            z2**2 * z1, z2**2 * z3,
            z3**2 * z1, z3**2 * z2,
            z1 * z2 * z3,
        ])


# =============================================================================
# ÖN İŞLEME FONKSİYONU — hattınızı buraya uygulayın
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ham öznitelik DataFrame'ine tüm ön işleme adımlarını uygular.

    Parametreler
    ------------
    df : pd.DataFrame
        X_test.csv'den yüklenen ham öznitelikler.

    Döndürür
    --------
    pd.DataFrame
        Model çıkarımı için hazır, ön işlenmiş öznitelikler.
        Çıkışın sütun isimleri ve sırası, girişle birebir aynıdır.
    """

    df = df.copy()

    # x1, x2, x3'ü eğitim parametreleriyle z-skor olarak standartlaştır
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(float)
    df[FEATURE_COLS] = (df[FEATURE_COLS].values - MEAN) / STD

    # x4_irrelevant olduğu gibi kalır; sütun isimleri/sırası korunur.
    # Modelin Pipeline'ı inference sırasında onu atacak.

    return df


# =============================================================================
# GİRİŞ NOKTASI — bu satırın altını değiştirmeyin
# =============================================================================

if __name__ == "__main__":
    GIRIS_DOSYASI  = "X_test.csv"
    CIKIS_DOSYASI  = "X_test_preprocessed.csv"

    print(f"{GIRIS_DOSYASI} okunuyor...")
    ham = pd.read_csv(GIRIS_DOSYASI)
    print(f"  Giriş boyutu : {ham.shape}")

    islenmis = preprocess(ham)
    print(f"  Çıkış boyutu : {islenmis.shape}")

    islenmis.to_csv(CIKIS_DOSYASI, index=False)
    print(f"{CIKIS_DOSYASI} kaydedildi.")
