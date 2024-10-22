# Machine-Learning-buat-yang-gak-jago-mtk
Diambil dari sini: https://www.youtube.com/watch?v=WH1SduDRL_Y&amp;t=9919s

## Outlier
semacam data yang berlebihan diisi , misal umur manusia ditentuin batasnya sampe 100 tapi inputnya bisa 200 nah itu outlier atau inacurate value, tapi di kasus misal nih harga mobil bisa outlier, sebenarnya gak juga karena mungkin harga mobilnya emang mahal banget.

## Saat analis data, akan ada namanya normalisasi data

**Normalisasi Data: Kunci Preprocessing di Machine Learning!**

Jadi, sebelum kita mulai nge-ML, penting banget buat normalisasi data. Ini adalah cara kita bikin data kita lebih siap tempur. Yuk, kita lihat beberapa teknik normalisasi yang bisa kamu coba:

1. **Min-Max Scaling**:
   - Ini adalah cara yang bikin semua nilai fitur kamu ada di rentang [0, 1]. Jadi, semua jadi seragam.
   - Rumusnya:
     \[
     X' = \frac{X - X_{min}}{X_{max} - X_{min}}
     \]

2. **Z-Score Normalization (Standardization)**:
   - Teknik ini bikin data kamu punya rata-rata 0 dan deviasi standar 1. Kayak semua data ngumpul di satu titik, gitu.
   - Rumus:
     \[
     X' = \frac{X - \mu}{\sigma}
     \]

3. **Robust Scaling**:
   - Kalo ada outlier yang nyeleneh, teknik ini pakai median dan rentang interkuartil (IQR) biar data kamu tetap oke.
   - Rumus:
     \[
     X' = \frac{X - \text{median}}{IQR}
     \]

4. **Log Transformation**:
   - Ini buat ngubah distribusi data yang miring jadi lebih normal. Jadi, kalo ada angka kecil yang bikin aneh, kita bisa atasi dengan log.
   - Rumus:
     \[
     X' = \log(X + 1)
     \]

5. **Power Transformation**:
   - Teknik ini bikin distribusi data kamu jadi lebih normal dengan fungsi pangkat atau akar. Misalnya, kamu bisa coba Box-Cox atau Yeo-Johnson.

6. **Decimal Scaling**:
   - Di sini, kamu bisa geser nilai desimal ke kiri. Jadi, semua angka dibagi dengan 10 pangkat \(j\).
   - Rumus:
     \[
     X' = \frac{X}{10^j}
     \]

**Ingat!** Pilih teknik yang paling cocok sama data kamu dan algoritma yang mau dipakai. Pastikan juga data latih dan data uji dinormalisasi dengan cara yang sama supaya hasilnya lebih akurat!

---

Gitu deh! Semoga ini membantu dan bikin kamu lebih semangat dalam belajar machine learning! 🚀
   - Di mana \(j\) adalah jumlah digit maksimum dari nilai yang akan dinormalisasi.

Pemilihan teknik normalisasi tergantung pada distribusi data dan algoritma machine learning yang digunakan. Pastikan untuk melakukan normalisasi pada data pelatihan dan data pengujian dengan cara yang sama!

### contoh 

Tentu! Berikut ini penjelasan tentang contoh penerapan teknik normalisasi data dengan gaya yang lebih santai:

---

**Contoh Normalisasi Data Pakai Python**

Yuk, kita coba beberapa teknik normalisasi data dengan Python! Kita bakal pakai `pandas` dan `scikit-learn`. Cekidot!

### Contoh Data

Kita mulai dengan bikin DataFrame sederhana:

```python
import pandas as pd

data = {
    'Fitur1': [10, 20, 30, 40, 50],
    'Fitur2': [100, 200, 300, 400, 500]
}

df = pd.DataFrame(data)
```

### 1. Min-Max Scaling

Kita ubah semua nilai ke rentang [0, 1]:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_min_max = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Min-Max Scaling:\n", df_min_max)
```

### 2. Z-Score Normalization

Sekarang, kita bikin data punya rata-rata 0 dan deviasi standar 1:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_standard = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Z-Score Normalization:\n", df_standard)
```

### 3. Robust Scaling

Kalau ada outlier yang nyeleneh, kita pakai teknik ini:

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df_robust = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Robust Scaling:\n", df_robust)
```

### 4. Log Transformation

Biar data lebih normal, kita pakai log transformation:

```python
import numpy as np

df_log = np.log1p(df)  # log(x + 1)

print("Log Transformation:\n", df_log)
```

### 5. Power Transformation (Box-Cox)

Bikin distribusi data lebih mendekati normal:

```python
from sklearn.preprocessing import PowerTransformer

scaler = PowerTransformer(method='box-cox')
df_boxcox = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

print("Box-Cox Transformation:\n", df_boxcox)
```

### 6. Decimal Scaling

Terakhir, kita geser desimal ke kiri:

```python
# Menentukan jumlah digit maksimum
j = 3  # Misalnya, kita bagi dengan 1000
df_decimal = df / (10 ** j)

print("Decimal Scaling:\n", df_decimal)
```

### Hasilnya

Setiap teknik bakal kasih hasil yang beda-beda, tergantung dari data yang kita punya. Pilih yang paling pas buat data dan algoritma yang mau kamu pakai!

---

Semoga ini bikin kamu lebih paham tentang normalisasi data dengan cara yang asyik! Selamat coding! 🚀
