import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# --- K-Medoids PAM fallback (karena scikit-learn-extra gagal build di environment ini) ---
from sklearn.metrics.pairwise import euclidean_distances


class KMedoidsPAM:
    def __init__(self, n_clusters, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_predict(self, X):
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        medoid_idx = np.sort(rng.choice(n, size=self.n_clusters, replace=False))
        for _ in range(self.max_iter):
            medoids = X[medoid_idx]
            D = euclidean_distances(X, medoids)
            labels = D.argmin(axis=1)
            new_medoid_idx = medoid_idx.copy()
            improved = False
            for m in range(self.n_clusters):
                member_pos = np.where(labels == m)[0]
                if len(member_pos) == 0:
                    continue
                sub = X[member_pos]
                Dm = euclidean_distances(sub, sub)
                total = Dm.sum(axis=1)
                best_local_pos = member_pos[np.argmin(total)]
                if best_local_pos not in medoid_idx:
                    new_medoid_idx[m] = best_local_pos
                    improved = True
            medoid_idx = np.sort(new_medoid_idx)
            if not improved:
                break
        self.medoid_indices_ = medoid_idx
        self.medoid_ = X[medoid_idx]
        D = euclidean_distances(X, self.medoid_)
        self.labels_ = D.argmin(axis=1)
        return self.labels_


# ================= TAHAP 6: Standardisasi Z-Score =================
INPUT = "lifesnaps_cleaned_for_clustering.csv"
OUTPUT_CLUSTERED = "lifesnaps_clustered_final.csv"
OUTPUT_EVAL = "tahap9_perbandingan_clustering.csv"

df = pd.read_csv(INPUT)
features = [
    'steps', 'sedentary_minutes', 'lightly_active_minutes',
    'moderately_active_minutes', 'very_active_minutes', 'calories',
    'minutesAsleep', 'minutesAwake', 'sleep_efficiency'
]
X = df[features].copy()

# pastikan numerik & isi NaN sisa dengan median
for c in features:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(X.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Tahap 6 selesai: X_scaled shape = {X_scaled.shape} (mean~0, std~1)")

# ================= TAHAP 8: Inisialisasi 5 Model =================
K = 3  # disesuaikan dari Tahap 7 (kondisi data: silhouette puncak K=2, tapi K=3 lebih interpretabel)
DBSCAN_EPS = 2.2   # dari k-distance graph (knee bulk)
DBSCAN_MIN_SAMPLES = 18  # 2 x 9 fitur

models = {
    'K-Means': KMeans(n_clusters=K, random_state=42, n_init=10),
    'K-Medoids': KMedoidsPAM(n_clusters=K, random_state=42),
    'AHC (Average Linkage / UPGMA)': AgglomerativeClustering(n_clusters=K, linkage='average'),
    'DBSCAN': DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES),
    'GMM': GaussianMixture(n_components=K, random_state=42)
}

# ================= TAHAP 9: Evaluasi & Komparasi =================
results = []
for name, model in models.items():
    labels = model.fit_predict(X_scaled)
    colname = f'cluster_{name}'
    df[colname] = labels

    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    noise_pct = (np.sum(labels == -1) / len(labels)) * 100

    if n_clusters > 1:
        sil = silhouette_score(X_scaled[mask], labels[mask])
        dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
        chi = calinski_harabasz_score(X_scaled[mask], labels[mask])
    else:
        sil, dbi, chi = np.nan, np.nan, np.nan

    results.append({
        'Metode': name,
        'Klaster': n_clusters,
        'Silhouette': round(sil, 4),
        'Davies-Bouldin': round(dbi, 4),
        'Calinski-Harabasz': round(chi, 2),
        'Noise (%)': round(noise_pct, 2)
    })

eval_df = pd.DataFrame(results)
print("\n=== PERBANDINGAN PERFORMA 5 ALGORITMA ===")
print(eval_df.to_string(index=False))

eval_df.to_csv(OUTPUT_EVAL, index=False)
df.to_csv(OUTPUT_CLUSTERED, index=False)
print(f"\nDisimpan: {OUTPUT_CLUSTERED} dan {OUTPUT_EVAL}")

# ================= TAHAP 10: Profiling rata-rata nilai asli (K-Means) =================
kmeans_col = 'cluster_K-Means'
print(f"\n=== PROFILING KLASTER ({kmeans_col} - SKALA ASLI) ===")
profile = df.groupby(kmeans_col)[features].mean().round(2)
profile['n'] = df.groupby(kmeans_col).size()
print(profile)

# Kategori risiko sedenter (berdasarkan K-Means)
def risiko_sedenter(row):
    steps = row['steps']
    sedent = row['sedentary_minutes']
    very_active = row['very_active_minutes']
    sleep_eff = row['sleep_efficiency']
    mins_asleep = row['minutesAsleep']
    if steps < 5000 and sedent > 700 and very_active < 30 and sleep_eff < 85:
        return 'Risiko Sedenter Tinggi'
    if (5000 <= steps < 9000) or sedent > 600 or mins_asleep < 420 or sleep_eff < 85:
        return 'Transisi / Deteksi Dini Sedenter'
    return 'Aktif & Tidur Sehat'

df['risiko_sedenter'] = df.apply(risiko_sedenter, axis=1)
print("\n=== DISTRIBUSI KATEGORI RISIKO (K-Means) ===")
print(df['risiko_sedenter'].value_counts())

df.to_csv(OUTPUT_CLUSTERED, index=False)
print(f"\nSelesai. File akhir: {OUTPUT_CLUSTERED}")
