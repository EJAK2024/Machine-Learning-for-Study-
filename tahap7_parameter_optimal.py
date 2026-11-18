import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import os

try:
    from sklearn_extra.cluster import KMedoids
    HAVE_KMEDOIDS = True
except Exception:
    HAVE_KMEDOIDS = False

INPUT = "lifesnaps_scaled.csv"
OUTDIR = "tahap7_output"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(INPUT)
id_cols = [c for c in ["id", "date"] if c in df.columns]
X = df.drop(columns=id_cols).values
feature_names = [c for c in df.columns if c not in id_cols]
n_features = X.shape[1]
print(f"Data: {X.shape[0]} baris x {X.shape[1]} fitur (sudah Z-score)")

# ---------- K = 2..8 untuk K-Means, K-Medoids, AHC, GMM ----------
K_RANGE = range(2, 9)
inertia = {}
silhouette = {}
bic = {}
aic = {}
medoids_inertia = {}

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertia[k] = km.inertia_
    silhouette[k] = silhouette_score(X, km.labels_)

    # K-Medoids sederhana (PAM build): ambil center KMeans, snap ke titik terdekat sebagai medoid
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    centers = km_tmp.cluster_centers_
    dm = euclidean_distances(centers, X)
    medoid_idx = dm.argmin(axis=1)
    medoids = X[medoid_idx]
    # biaya total = jarak setiap titik ke medoid terdekat
    d2 = euclidean_distances(X, medoids)
    medoids_inertia[k] = float(np.sum(np.min(d2, axis=1)))

    ahc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    silhouette[k]  # AHC pakai silhouette sama; simpan terpisah jika perlu

    gmm = GaussianMixture(n_components=k, random_state=42).fit(X)
    bic[k] = gmm.bic(X)
    aic[k] = gmm.aic(X)

# ---------- Plot: Elbow (Inertia/WCSS) ----------
plt.figure(figsize=(8, 5))
plt.plot(list(K_RANGE), [inertia[k] for k in K_RANGE], "o-", color="tab:blue")
plt.xlabel("Jumlah klaster K")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method - K-Means")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/elbow_kmeans.png", dpi=150)
plt.close()

# ---------- Plot: Silhouette ----------
plt.figure(figsize=(8, 5))
plt.plot(list(K_RANGE), [silhouette[k] for k in K_RANGE], "o-", color="tab:green")
plt.xlabel("Jumlah klaster K")
plt.ylabel("Skor Silhouette")
plt.title("Silhouette Analysis (K=2..8)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/silhouette.png", dpi=150)
plt.close()

# ---------- Plot: BIC / AIC untuk GMM ----------
plt.figure(figsize=(8, 5))
plt.plot(list(K_RANGE), [bic[k] for k in K_RANGE], "o-", label="BIC", color="tab:red")
plt.plot(list(K_RANGE), [aic[k] for k in K_RANGE], "o-", label="AIC", color="tab:orange")
plt.xlabel("Jumlah komponen K")
plt.ylabel("Skor")
plt.title("BIC / AIC - GMM (semakin kecil semakin baik)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/bic_aic_gmm.png", dpi=150)
plt.close()

# ---------- Dendrogram AHC (Ward) ----------
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8,
           color_threshold=0, above_threshold_color="tab:gray")
plt.title("Dendrogram - AHC Ward's Linkage (potongan = jumlah klaster)")
plt.xlabel("Indeks sampel (truncated)")
plt.ylabel("Jarak")
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/dendrogram_ahc.png", dpi=150)
plt.close()

# ---------- DBSCAN: k-distance graph (k = 2 * n_fiturs = 18) ----------
from sklearn.neighbors import NearestNeighbors
k_db = 2 * n_features  # 18
nbrs = NearestNeighbors(n_neighbors=k_db).fit(X)
distances, indices = nbrs.kneighbors(X)
k_dist = np.sort(distances[:, k_db - 1])  # jarak ke tetangga ke-k_db

plt.figure(figsize=(8, 5))
plt.plot(k_dist, color="tab:purple")
plt.axhline(y=k_dist[int(len(k_dist) * 0.9)], color="tab:red", linestyle="--",
            label=f"90th percentile ≈ {k_dist[int(len(k_dist)*0.9)]:.3f}")
plt.xlabel("Titik data (diurutkan)")
plt.ylabel(f"Jarak ke tetangga ke-{k_db}")
plt.title(f"k-Distance Graph (k = {k_db} = 2×{n_features} fitur) untuk DBSCAN eps")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/kdistance_dbscan.png", dpi=150)
plt.close()

# ---------- Ringkasan parameter yang disarankan ----------
best_k_sil = max(silhouette, key=silhouette.get)
best_k_bic = min(bic, key=bic.get)
best_k_aic = min(aic, key=aic.get)

# Knee sederhana untuk eps: ambil titik dengan kemiringan terbesar (maximal diff)
diffs = np.diff(k_dist)
knee_idx = int(np.argmax(diffs)) + 1
eps_suggested = k_dist[knee_idx]

print("\n=== REKOMENDASI PARAMETER ===")
print(f"K-Means      : K terbaik (silhouette) = {best_k_sil}  [sil={silhouette[best_k_sil]:.4f}]")
print(f"K-Medoids    : K disarankan = {best_k_sil}  (elbow cost K-Medoids dihitung via PAM-build)")
print(f"AHC (Ward)   : K terbaik = {best_k_sil}  (lihat dendrogram untuk potongan visual)")
print(f"GMM          : K terbaik BIC = {best_k_bic}, AIC = {best_k_aic}")
print(f"DBSCAN       : min_samples = {k_db}  (2 x {n_features} fitur), eps disarankan = {eps_suggested:.3f} (knee k-distance)")

# Simpan ringkasan ke file
summary = pd.DataFrame([
    {"algoritma": "K-Means", "parameter": "K", "nilai": best_k_sil, "alas": "silhouette tertinggi"},
    {"algoritma": "K-Medoids", "parameter": "K", "nilai": best_k_sil, "alas": "elbow cost PAM-build / sama dengan K-Means"},
    {"algoritma": "AHC", "parameter": "K", "nilai": best_k_sil, "alas": "silhouette tertinggi / potong dendrogram"},
    {"algoritma": "GMM", "parameter": "K (BIC)", "nilai": best_k_bic, "alas": "BIC terkecil"},
    {"algoritma": "GMM", "parameter": "K (AIC)", "nilai": best_k_aic, "alas": "AIC terkecil"},
    {"algoritma": "DBSCAN", "parameter": "min_samples", "nilai": k_db, "alas": "2 x jumlah fitur"},
    {"algoritma": "DBSCAN", "parameter": "eps", "nilai": round(float(eps_suggested), 4), "alas": "knee k-distance graph"},
])
summary.to_csv(f"{OUTDIR}/parameter_rekomendasi.csv", index=False)
print(f"\nGrafik disimpan di folder '{OUTDIR}/' dan parameter_rekomendasi.csv")
