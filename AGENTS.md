# AGENTS.md — csv_rais_anonymized (PEMSIN clustering assignment)

Single-folder Python course project. **No build system, no package manifest, no git, no tests, no lint, no CI.** Every script is a standalone entrypoint using relative paths — run from this folder: `python <file>.py`.

## Kasus & metode
Judul: *Pengelompokan Pola Aktivitas Fisik dan Kualitas Tidur Pengguna Perangkat Wearable Menggunakan Algoritma K-Means untuk Deteksi Dini Gaya Hidup Sedenter.*

5 metode clustering (semua **unsupervised learning**, tidak memerlukan label):
1. K-Means (centroid-based)
2. K-Medoids / PAM (medoid-based)
3. AHC — Average Linkage / UPGMA (hierarchical)
4. DBSCAN (density-based)
5. GMM — Gaussian Mixture Model (probabilistic)

**Jawaban:** Ya, kasus ini dapat diselesaikan dengan unsupervised learning karena tidak ada ground-truth label "sedentary/aktif". Cluster yang ditemukan kemudian diberi interpretasi risiko sedenter secara aturan (`risiko_sedenter`). Cocok dijalankan di Jupyter Notebook.

## Commands
- Python 3.14.7; pandas, numpy, scikit-learn, matplotlib, scipy, fpdf, nbformat semua importable global. Tanpa `requirements.txt`.
- Tidak ada test runner.
- Notebook `program_clustering.ipynb` di-generate oleh `make_notebook.py`; regenerasi via `python make_notebook.py` (jangan edit `.ipynb` secara manual).
- Jalankan semua dari folder ini: `cd` ke folder dulu, lalu `python <file>.py`.

## Main entrypoint
`tahap6_10_gabungan.py` — gabungan Tahap 6–10. Baca `lifesnaps_cleaned_for_clustering.csv`, lakukan Z-score scaling, fit 5 model clustering, evaluasi, profiling, tulis:
- `lifesnaps_clustered_final.csv` (asli + 5 kolom label + `risiko_sedenter`)
- `tahap9_perbandingan_clustering.csv` (Silhouette / Davies-Bouldin / Calinski-Harabasz / noise per metode)

## Pipeline order & data
1. `daily_fitbit_sema_df_unprocessed.csv` (7,410 baris mentah) → `tahap3_pembersihan_data.py` atau `panda as.py` → `lifesnaps_cleaned_for_clustering.csv` (~3,529 baris × 9 fitur setelah dropna + filter anomali sensor)
2. `tahap_standardisasi.py` → `lifesnaps_scaled.csv` (StandardScaler; `id`/`date` dikecualikan)
3. `tahap7_parameter_optimal.py` baca `lifesnaps_scaled.csv` → `tahap7_output/` (elbow, silhouette, BIC/AIC, dendrogram, k-distance + `parameter_rekomendasi.csv`; `matplotlib.use("Agg")`, headless-safe)
4. `tahap8_eksekusi_clustering.py` baca `lifesnaps_scaled.csv` → `lifesnaps_clustered.csv`
5. `tahap9_evaluasi.py` baca `lifesnaps_clustered.csv` → `tahap9_perbandingan_clustering.csv`
6. `tahap10_interpretasi.py` baca cleaned + clustered → `tahap10_output/` (PCA, radar, bar plot, `lifesnaps_with_risiko.csv`)

Fitur: `steps`, `sedentary_minutes`, `lightly_active_minutes`, `moderately_active_minutes`, `very_active_minutes`, `calories`, `minutesAsleep`, `minutesAwake`, `sleep_efficiency`.

## Quirks / gotchas
- `sklearn_extra.cluster.KMedoids` gagal build di environment ini → diganti class `KMedoidsPAM` custom (`tahap6_10_gabungan.py:13`, `tahap8_eksekusi_clustering.py:19`).
- AHC pakai `linkage='average'` (UPGMA), bukan ward (`tahap6_10_gabungan.py:80`).
- DBSCAN `eps=2.2` (12.579 membuat 1 klaster); `min_samples = 2 × 9 fitur = 18`.
- `K=3` dipilih walau silhouette puncak di `K=2` (lebih interpretabel).
- `program_clustering.py` interaktif (`input()`), tidak bisa headless; AHC >300 baris meminta konfirmasi (O(n³)).
- Semua CSV pakai path relatif; `cd` ke folder sebelum menjalankan.
- File yang tidak digunakan dipindahkan ke `arsip_tidak_digunakan/` (termasuk `tahap4_clustering.py`, `buat_pdf_dokumentasi.py`, PDF dokumen, `__pycache__`, notebook hasil generate, dan berkas referensi lain).
