import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from math import pi

ORIG = "lifesnaps_cleaned_for_clustering.csv"   # data asli (belum scaling)
CLUST = "lifesnaps_clustered.csv"               # scaled + label klaster
OUTDIR = "tahap10_output"
import os
os.makedirs(OUTDIR, exist_ok=True)

df_orig = pd.read_csv(ORIG)
df_clust = pd.read_csv(CLUST)

id_cols = [c for c in ["id", "date"] if c in df_clust.columns]
label_cols = ["cluster_kmeans", "cluster_kmedoids", "cluster_ahc", "cluster_dbscan", "cluster_gmm"]

# Gabungkan label klaster ke data asli (unscaled) berdasarkan id+date
df = df_orig.merge(df_clust[id_cols + label_cols], on=id_cols, how="inner")
print(f"Data tergabung: {df.shape}")

# ---------- 1. Rata-rata nilai ASLI per klaster ----------
feature_cols = [c for c in df_orig.columns if c not in id_cols]

def cluster_profile(data, label_col):
    prof = data.groupby(label_col)[feature_cols].mean()
    prof["n"] = data.groupby(label_col).size()
    return prof

print("\n=== PROFIL KLASTER (nilai asli, sebelum scaling) ===")
for lc in label_cols:
    print(f"\n--- {lc} ---")
    print(cluster_profile(df, lc).round(2))

# ---------- 2. Label kategori risiko sedenter ----------
def risiko_sedenter(row):
    steps = row["steps"]
    sedent = row["sedentary_minutes"]
    very_active = row["very_active_minutes"]
    sleep_eff = row["sleep_efficiency"]
    mins_asleep = row["minutesAsleep"]
    if steps < 5000 and sedent > 700 and very_active < 30 and sleep_eff < 85:
        return "Risiko Sedenter Tinggi"
    if (5000 <= steps < 9000) or sedent > 600 or mins_asleep < 420 or sleep_eff < 85:
        return "Transisi / Deteksi Dini Sedenter"
    return "Aktif & Tidur Sehat"

df["risiko_kmeans"] = df.apply(risiko_sedenter, axis=1)
df["risiko_dbscan"] = df.apply(risiko_sedenter, axis=1)

print("\n=== DISTRIBUSI KATEGORI RISIKO (K-Means) ===")
print(df["risiko_kmeans"].value_counts())
print("\n=== DISTRIBUSI KATEGORI RISIKO (DBSCAN) ===")
print(df["risiko_dbscan"].value_counts())

# Simpan data lengkap dengan label risiko
df.to_csv(f"{OUTDIR}/lifesnaps_with_risiko.csv", index=False)

# ---------- 3. Visualisasi: PCA 2D scatter untuk 5 metode ----------
X_scaled = df_clust.drop(columns=id_cols).values
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharex=True, sharey=True)
cmap = plt.get_cmap("tab10")
for ax, lc in zip(axes, label_cols):
    labels = df_clust[lc].values
    uniq = np.unique(labels)
    for i, lab in enumerate(uniq):
        mask = labels == lab
        lbl = f"Noise (-1)" if lab == -1 else f"Klaster {lab}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[cmap(i % 10)], s=12, alpha=0.6, edgecolors="none", label=lbl)
    ax.set_title(lc.replace("cluster_", "").upper() + f"\n(k={len(uniq)})", fontsize=10)
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.legend(fontsize=6, loc="best", framealpha=0.8)
axes[0].set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
fig.suptitle("PCA 2D - Sebaran Klaster 5 Metode (data Z-score)", fontsize=13)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(f"{OUTDIR}/pca_2d_five_methods.png", dpi=150)
plt.close(fig)

# ---------- 4. Radar chart profil klaster (K-Means & DBSCAN) ----------
def radar_profile(data, label_col, title, fname):
    prof = data.groupby(label_col)[feature_cols].mean()
    # Normalisasi min-max per fitur untuk radar (0..1)
    prof_norm = (prof - prof.min()) / (prof.max() - prof.min() + 1e-9)
    n_feat = len(feature_cols)
    angles = [n / float(n_feat) * 2 * pi for n in range(n_feat)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.get_cmap("tab10").colors
    for i, (cl, row) in enumerate(prof_norm.iterrows()):
        vals = row.tolist()
        vals += vals[:1]
        lab = "Noise" if cl == -1 else f"Klaster {cl}"
        ax.plot(angles, vals, color=colors[i % 10], linewidth=2, label=lab)
        ax.fill(angles, vals, color=colors[i % 10], alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_cols, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

radar_profile(df, "cluster_kmeans",
              "Profil Klaster - K-Means (nilai asli, min-max norm)",
              f"{OUTDIR}/radar_kmeans.png")
radar_profile(df, "cluster_dbscan",
              "Profil Klaster - DBSCAN (nilai asli, min-max norm)",
              f"{OUTDIR}/radar_dbscan.png")

# ---------- 5. Bar plot profil klaster ----------
def bar_profile(data, label_col, title, fname):
    prof = data.groupby(label_col)[feature_cols].mean()
    prof_plot = prof.drop(columns=["n"]) if "n" in prof.columns else prof
    ax = prof_plot.plot(kind="bar", figsize=(12, 5))
    ax.set_title(title)
    ax.set_ylabel("Rata-rata nilai asli")
    ax.set_xlabel("Klaster")
    ax.legend(title="Fitur", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.tick_params(axis="x", rotation=0)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)

bar_profile(df, "cluster_kmeans", "Profil Klaster - K-Means (rata-rata nilai asli)",
            f"{OUTDIR}/bar_kmeans.png")
bar_profile(df, "cluster_dbscan", "Profil Klaster - DBSCAN (rata-rata nilai asli)",
            f"{OUTDIR}/bar_dbscan.png")

print(f"\nSemua output disimpan di folder '{OUTDIR}/'")
print("Selesai Tahap 10.")
