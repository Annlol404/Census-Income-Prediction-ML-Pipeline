# Census Income Prediction

Prediksi kategori pendapatan individu (`<=50K` atau `>50K` USD/tahun) menggunakan Census Income (KDD) Dataset. Proyek ini dikerjakan sebagai bagian dari seleksi **Study Group AI Engineering - Laboratorium Big Data, Universitas Telkom**.

dengan F1-macro 0,82 pada validation set (naik dari 0,80 setelah hyperparameter tuning dan threshold tuning yang divalidasi dengan 5-fold cross-validation).

## Ringkasan Proyek

| | |
|---|---|
| **Task** | Klasifikasi biner |
| **Dataset** | Adult / Census Income (KDD), 39.073 baris (train), 9.769 baris (test) |
| **Model final** | XGBoost (tuned) |
| **Metrik evaluasi** | F1-macro |
| **Skor validasi** | 0,8192 |
| **Skor leaderboard** | Peringkat #2 |

## Struktur Repository

```
census-income-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── census_income_pipeline.ipynb   # Pipeline end-to-end: EDA → preprocessing → FE → modeling → evaluasi → tuning
├── submissionsFinal.cvs     # Submission final setelah tuning (F1-macro 0,82)
```

## Pipeline

1. **Exploratory Data Analysis (EDA)** : struktur data, distribusi target, missing value tersembunyi (`NaN` + `"?"`), duplikat, pola hubungan fitur-target.
2. **Data Preprocessing** : normalisasi label target (4 varian → 2 kelas), imputasi missing value dengan kategori `"Unknown"`, drop duplikat.
3. **Feature Engineering** : drop `fnlwgt`/`education` (redundan), fitur turunan `has_capital_gain`, `has_capital_loss`, `is_us`.
4. **Modeling** : perbandingan Logistic Regression, Random Forest, XGBoost (semua dengan mekanisme penyeimbang kelas untuk data imbalanced 76:24).
5. **Evaluasi** : F1-macro, classification report, confusion matrix, feature importance.
6. **Model Improvement** : hyperparameter tuning (RandomizedSearchCV) + threshold tuning (divalidasi 5-fold cross-validation) → F1-macro naik dari 0,80 ke 0,82.
7. **Analisis Tambahan (Soal Bonus)** : kelompok pekerjaan/pendidikan dengan proporsi `>50K` tertinggi, pemeriksaan kualitas data.

## Temuan Kunci

- **Prediktor paling berpengaruh:** status pernikahan (`Married-civ-spouse`), `capital-gain`, dan jenjang pendidikan (`education-num`).
- **Kelompok pekerjaan** dengan proporsi `>50K` tertinggi: `Exec-managerial` (47,25%).
- **Tingkat pendidikan** dengan proporsi `>50K` tertinggi: `Prof-school` (74,92%) dan `Doctorate` (74,57%).
- **Threshold tuning** (menggeser ambang keputusan dari 0,5 ke 0,70) memberikan peningkatan performa paling signifikan dibanding hyperparameter tuning maupun fitur tambahan.

## Cara Menjalankan

```bash
# Clone repo
git clone https://github.com/<username>/census-income-prediction.git
cd census-income-prediction

# Install dependencies
pip install -r requirements.txt

# Letakkan train.csv, test.csv, sample_submission.csv di folder data/
# (dataset tidak disertakan di repo ini karena ketentuan lomba)

# Jalankan notebook
jupyter notebook notebooks/census_income_pipeline.ipynb
```

## Data

Dataset (`train.csv`, `test.csv`, `sample_submission.csv`) disediakan oleh panitia Laboratorium Big Data Universitas Telkom dan tidak disertakan di repository ini. Dataset serupa (Adult/Census Income) tersedia secara publik di [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult).

## Tools

Python · pandas · numpy · scikit-learn · XGBoost · matplotlib · seaborn · Jupyter Notebook

## Lisensi

Proyek ini dibuat untuk keperluan seleksi akademik (Study Group AI Engineering, Laboratorium Big Data Universitas Telkom) dan bersifat non-komersial.
