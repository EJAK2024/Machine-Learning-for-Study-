import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

INPUT_X = "lifesnaps_scaled.csv"
INPUT_LABELS = "lifesnaps_clustered.csv"
OUTPUT = "tahap9_perbandingan_clustering.csv"

df = pd.read_csv(INPUT_LABELS)
id_cols = [c for c in ["id", "date"] if c in df.columns]
X = df.drop(columns=id_cols).values

methods = ["cluster_kmeans", "cluster_kmedoids", "cluster_ahc", "cluster_dbscan", "cluster_gmm"]
rows = []

for col in methods:
    labels = df[col].values
    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())
    pct_noise = 100.0 * n_noise / len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Untuk metrik: gunakan semua titik non-noise (DBSCAN); untuk non-DBSCAN semua titik
    if n_noise > 0:
        Xm = X[~noise_mask]
        lm = labels[~noise_mask]
    else:
        Xm = X
        lm = labels

    uniq = np.unique(lm)
    valid = len(uniq) > 1

    try:
        sil = silhouette_score(Xm, lm) if valid else np.nan
    except Exception:
        sil = np.nan
    try:
        dbi = davies_bouldin_score(Xm, lm) if valid else np.nan
    except Exception:
        dbi = np.nan
    try:
        ch = calinski_harabasz_score(Xm, lm) if valid else np.nan
    except Exception:
        ch = np.nan

    rows.append({
        "Metode Clustering": col.replace("cluster_", "").upper(),
        "Jumlah Klaster Terbentuk": n_clusters,
        "Silhouette Score": round(sil, 4) if valid else np.nan,
        "Davies-Bouldin Index": round(dbi, 4) if valid else np.nan,
        "Calinski-Harabasz Index": round(ch, 2) if valid else np.nan,
        "Persentase Noise": round(pct_noise, 2),
    })

res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv(OUTPUT, index=False)
print(f"\nDisimpan: {OUTPUT}")

# Menentukan yang paling unggul (berdasarkan Silhouette, dan DBI rendah, CH tinggi, noise rendah)
print("\nRekomendasi metode terbaik:")
best_sil = res.loc[res["Silhouette Score"].idxmax()]
print(f"- Silhouette tertinggi: {best_sil['Metode Clustering']} ({best_sil['Silhouette Score']})")
