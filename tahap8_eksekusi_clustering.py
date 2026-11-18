import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import euclidean_distances
import os

INPUT = "lifesnaps_scaled.csv"
OUTPUT = "lifesnaps_clustered.csv"
RANDOM_STATE = 42

# Parameter optimal dari Tahap 7
K_PARTITION = 2   # K-Means, K-Medoids, AHC (silhouette tertinggi)
K_GMM = 6         # GMM (BIC/AIC terkecil)
DBSCAN_EPS = 2.2  # koreksi: knee k-distance bulk ≈ persentil-90 (~1.98); 12.579 adalah outlier maks (terlalu besar -> 1 klaster)
DBSCAN_MIN_SAMPLES = 18  # 2 x 9 fitur


class KMedoidsPAM:
    """K-Medoids PAM (Partitioning Around Medoids), build fase."""
    def __init__(self, n_clusters, max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        # Inisialisasi medoid acak
        medoid_idx = np.sort(rng.choice(n, size=self.n_clusters, replace=False))
        for _ in range(self.max_iter):
            medoids = X[medoid_idx]
            # Assign setiap titik ke medoid terdekat
            D = euclidean_distances(X, medoids)
            labels = D.argmin(axis=1)
            new_medoid_idx = medoid_idx.copy()
            improved = False
            # Swap fase: coba ganti setiap medoid dengan anggota clusternya
            for m in range(self.n_clusters):
                member_pos = np.where(labels == m)[0]
                if len(member_pos) == 0:
                    continue
                # candidate medoid = anggota cluster m yang meminimalkan total jarak ke anggota cluster
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
        return self


df = pd.read_csv(INPUT)
id_cols = [c for c in ["id", "date"] if c in df.columns]
X = df.drop(columns=id_cols).values
print(f"Data siap model: {X.shape[0]} baris x {X.shape[1]} fitur (Z-score)")

# --- 1. K-Means ---
km = KMeans(n_clusters=K_PARTITION, random_state=RANDOM_STATE, n_init=10)
df["cluster_kmeans"] = km.fit_predict(X)

# --- 2. K-Medoids (PAM) ---
kmed = KMedoidsPAM(n_clusters=K_PARTITION, random_state=RANDOM_STATE)
kmed.fit(X)
df["cluster_kmedoids"] = kmed.labels_

# --- 3. AHC (Ward's Linkage) ---
ahc = AgglomerativeClustering(n_clusters=K_PARTITION, linkage="ward")
df["cluster_ahc"] = ahc.fit_predict(X)

# --- 4. DBSCAN ---
dbs = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
df["cluster_dbscan"] = dbs.fit_predict(X)

# --- 5. GMM ---
gmm = GaussianMixture(n_components=K_GMM, random_state=RANDOM_STATE)
df["cluster_gmm"] = gmm.fit_predict(X)

df.to_csv(OUTPUT, index=False)

print("\nDistribusi label per algoritma:")
for col in ["cluster_kmeans", "cluster_kmedoids", "cluster_ahc", "cluster_dbscan", "cluster_gmm"]:
    vc = df[col].value_counts().sort_index()
    print(f"  {col}: {dict(vc)}  (n_klaster={df[col].nunique()})")

print(f"\nDisimpan: {OUTPUT}  ({df.shape})")
print("Kolom label ditambahkan:", ["cluster_kmeans", "cluster_kmedoids", "cluster_ahc", "cluster_dbscan", "cluster_gmm"])
