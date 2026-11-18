import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score

INPUT = "daily_fitkit_sema_df_clean.csv"  # fallback jika nama berbeda
import os
CANDIDATES = ["daily_fitbit_sema_df_clean.csv", "lifesnaps_cleaned_for_clustering.csv"]
INPUT = next((c for c in CANDIDATES if os.path.exists(c)), None)
if INPUT is None:
    raise FileNotFoundError("Tidak ditemukan file hasil pembersihan (daily_fitbit_sema_df_clean.csv / lifesnaps_cleaned_for_clustering.csv)")

OUTPUT_CLEAN = "lifesnaps_cleaned_for_clustering.csv"
OUTPUT_LABEL = "lifesnaps_cluster_labels.csv"
N_CLUSTERS = 4
RANDOM_STATE = 42

df = pd.read_csv(INPUT)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.to_csv(OUTPUT_CLEAN, index=False)
print(f"[1] Disimpan {OUTPUT_CLEAN}: {df.shape}")

# --- Pilih fitur numerik inti untuk clustering (aktivitas + tidur), tanpa id/date/kategorikal/one-hot ---
FEATURES = [
    "steps", "sedentary_minutes", "calories", "distance", "bpm",
    "lightly_active_minutes", "moderately_active_minutes", "very_active_minutes",
    "minutesAsleep", "sleep_efficiency", "sleep_duration_min",
    "minutesToFallAsleep", "minutesAwake", "stress_score", "resting_hr",
    "nremhr", "rmssd",
]
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Fitur tidak ditemukan: {missing}")

X = df[FEATURES].copy()
# pastikan numerik & isi NaN sisa dengan median (aman setelah dropna tahap 3, tetapi jaga-jaga)
for c in FEATURES:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(X.median(numeric_only=True))

# --- Standardisasi Z-Score (StandardScaler) ---
scaler = StandardScaler()
Xz = scaler.fit_transform(X)
Xz = pd.DataFrame(Xz, columns=FEATURES, index=X.index)
print(f"[2] StandardScaler selesai. Xz shape: {Xz.shape}")

def eval_model(name, labels, Xmat):
    # hanya hitung metrik jika ada >= 2 klaster dan tidak semua noise (-1)
    uniq = np.unique(labels)
    valid = len(uniq) > 1 and not (len(uniq) == 2 and -1 in uniq)
    try:
        sil = silhouette_score(Xmat, labels) if valid else np.nan
    except Exception:
        sil = np.nan
    try:
        ch = calinski_harabasz_score(Xmat, labels) if valid else np.nan
    except Exception:
        ch = np.nan
    n_noise = int((labels == -1).sum()) if -1 in uniq else 0
    return {
        "model": name, "n_clusters": len(uniq) - (1 if -1 in uniq else 0),
        "n_noise": n_noise, "silhouette": sil, "calinski_harabasz": ch,
    }

results = []
out_labels = pd.DataFrame(index=df.index)
out_labels["id"] = df["id"].values if "id" in df.columns else np.arange(len(df))
out_labels["date"] = df["date"].values if "date" in df.columns else ""

# 1. K-Means
km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
labels = km.fit_predict(Xz)
out_labels["kmeans"] = labels
results.append(eval_model("KMeans", labels, Xz))

# 2. Agglomerative Clustering (Hierarkis)
agg = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
labels = agg.fit_predict(Xz)
out_labels["agglomerative"] = labels
results.append(eval_model("Agglomerative", labels, Xz))

# 3. DBSCAN
dbs = DBSCAN(eps=2.0, min_samples=10)
labels = dbs.fit_predict(Xz)
out_labels["dbscan"] = labels
results.append(eval_model("DBSCAN", labels, Xz))

# 4. Gaussian Mixture Model
gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=RANDOM_STATE)
labels = gmm.fit_predict(Xz)
out_labels["gmm"] = labels
results.append(eval_model("GMM", labels, Xz))

# 5. Spectral Clustering
sp = SpectralClustering(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, assign_labels="kmeans", n_init=10)
labels = sp.fit_predict(Xz)
out_labels["spectral"] = labels
results.append(eval_model("Spectral", labels, Xz))

out_labels.to_csv(OUTPUT_LABEL, index=False)
res_df = pd.DataFrame(results)
print("\n[3] Hasil 5 algoritma clustering (pada X z-score):")
print(res_df.to_string(index=False))
res_df.to_csv("clustering_comparison.csv", index=False)
print(f"\nDisimpan: {OUTPUT_LABEL}, clustering_comparison.csv")
print("Selesai.")
