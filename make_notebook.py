import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

nb = nbf.v4.new_notebook()

nb.cells = [
    new_markdown_cell("""# Program Clustering — lifesnaps_cleaned_for_clustering.csv

Notebook ini membaca file `lifesnaps_cleaned_for_clustering.csv`, lalu menjalankan perhitungan clustering pada data yang dipilih user.

**Alur:**
1. Load semua data dari file CSV.
2. Pilih metode clustering (default: **AHC Average Linkage / UPGMA**).
3. Pilih indeks baris data yang akan dihitung (atau semua).
4. Program menghasilkan matriks jarak, linkage matrix, hasil cluster per baris, anggota setiap cluster, dan silhouette score.

Ubah variabel di sel Pengaturan lalu jalankan sel-selnya berurutan (Kernel → Restart & Run All)."""),

    new_code_cell("""import numpy as np
import pandas as pd
from IPython.display import display, HTML

CSV_PATH = "lifesnaps_cleaned_for_clustering.csv"
NUMERIC_COLS = [
    "steps", "sedentary_minutes", "lightly_active_minutes",
    "moderately_active_minutes", "very_active_minutes",
    "calories", "minutesAsleep", "minutesAwake", "sleep_efficiency"
]

# === PENGATURAN USER (ubah di sini) ===
METODE = "1"          # "1" = AHC Average Linkage (UPGMA), "2" = K-Means, "3" = DBSCAN
PILIH_INDEKS = None   # None = semua data; atau daftar misal [0, 1, 2, 3, 4]
K_CLUSTER = 3         # jumlah cluster untuk AHC / K-Means
# =====================================

df = pd.read_csv(CSV_PATH)
print(f"Data dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
print("Kolom numerik:", NUMERIC_COLS)
display(df.head(10))"""),

    new_code_cell("""def jarak_euclid(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))

def ahc_average_linkage(X):
    # Agglomerative Hierarchical Clustering dengan Average Linkage (UPGMA).
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = jarak_euclid(X[i], X[j])
            D[j, i] = D[i, j]

    members = {i: {i} for i in range(n)}
    active = list(range(n))
    next_id = n
    linkage = []

    while len(active) > 1:
        best = None
        best_d = np.inf
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                c1, c2 = active[a], active[b]
                m1, m2 = members[c1], members[c2]
                ds = [D[i][j] for i in m1 for j in m2]
                d = float(np.mean(ds))
                if d < best_d:
                    best_d = d
                    best = (c1, c2)
        c1, c2 = best
        size = len(members[c1]) + len(members[c2])
        linkage.append([float(c1), float(c2), best_d, float(size)])
        new_id = next_id
        next_id += 1
        members[new_id] = members[c1] | members[c2]
        active.remove(c1)
        active.remove(c2)
        active.append(new_id)

    return np.array(linkage), D

def potong_dendrogram(linkage, n_asli, k):
    parent = list(range(n_asli + linkage.shape[0]))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for step in range(n_asli - k):
        c1 = int(linkage[step, 0])
        c2 = int(linkage[step, 1])
        r1, r2 = find(c1), find(c2)
        if r1 != r2:
            parent[r2] = r1
    root_to_label = {}
    labels = []
    for i in range(n_asli):
        r = find(i)
        if r not in root_to_label:
            root_to_label[r] = len(root_to_label)
        labels.append(root_to_label[r])
    return np.array(labels)

def hitung_silhouette(X, labels):
    n = X.shape[0]
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    s = []
    for i in range(n):
        own = labels[i]
        own_mask = labels == own
        own_mask[i] = False
        a = float(np.mean([np.linalg.norm(X[i] - X[j]) for j in np.where(own_mask)[0]])) if own_mask.any() else 0.0
        b = np.inf
        for c in uniq:
            if c == own:
                continue
            cm = labels == c
            dist = float(np.mean([np.linalg.norm(X[i] - X[j]) for j in np.where(cm)[0]]))
            if dist < b:
                b = dist
        s.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(s))"""),

    new_code_cell("""# Pilih baris data
if PILIH_INDEKS is None:
    idx = list(df.index)
else:
    idx = [i for i in PILIH_INDEKS if i in df.index]

sub = df.loc[idx].copy()
X = sub[NUMERIC_COLS].to_numpy(dtype=float)

print(f"Jumlah data yang akan dihitung: {len(idx)}")
display(sub.head(10))

if len(idx) > 300 and METODE == "1":
    display(HTML("<b>Peringatan:</b> AHC dengan banyak data akan lambat (O(n³)). Pertimbangkan pilih subset atau gunakan K-Means."))"""),

    new_code_cell("""# Jalankan perhitungan
if METODE == "1":
    k = max(2, min(K_CLUSTER, len(idx)))
    linkage, D = ahc_average_linkage(X)
    labels = potong_dendrogram(linkage, len(idx), k)

    display(HTML("<h3>Hasil AHC — Average Linkage (UPGMA)</h3>"))
    print(f"Jumlah cluster (k): {k}")

    display(HTML("<h4>Matriks Jarak Euclidean</h4>"))
    np.set_printoptions(precision=3, suppress=True)
    display(pd.DataFrame(np.round(D, 3), index=idx, columns=idx))

    display(HTML("<h4>Linkage Matrix</h4>"))
    print("Kolom: [cluster1, cluster2, jarak, ukuran_cluster_baru]")
    display(pd.DataFrame(linkage, columns=["c1", "c2", "jarak", "ukuran"]))

    display(HTML("<h4>Hasil Cluster per Baris</h4>"))
    hasil = pd.DataFrame({"indeks_asli": idx, "cluster": labels})
    for c in NUMERIC_COLS:
        hasil[c] = sub[c].values
    display(hasil)

    display(HTML("<h4>Anggota Setiap Cluster</h4>"))
    for c in sorted(np.unique(labels)):
        anggota = [str(i) for i, l in zip(idx, labels) if l == c]
        print(f"Cluster {c} ({len(anggota)} data): {', '.join(anggota)}")

    sil = hitung_silhouette(X, labels)
    print(f"\nSilhouette score (k={k}): {sil:.4f}")

elif METODE == "2":
    from sklearn.cluster import KMeans
    k = max(2, min(K_CLUSTER, len(idx)))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    display(HTML(f"<h3>Hasil K-Means (k={k}, inersia={km.inertia_:.4f})</h3>"))
    hasil = pd.DataFrame({"indeks_asli": idx, "cluster": labels})
    for c in NUMERIC_COLS:
        hasil[c] = sub[c].values
    display(hasil)
    for c in sorted(np.unique(labels)):
        anggota = [str(i) for i, l in zip(idx, labels) if l == c]
        print(f"Cluster {c} ({len(anggota)} data): {', '.join(anggota)}")

elif METODE == "3":
    from sklearn.cluster import DBSCAN
    eps = 0.5
    min_s = 5
    db = DBSCAN(eps=eps, min_samples=min_s)
    labels = db.fit_predict(X)
    display(HTML(f"<h3>Hasil DBSCAN (eps={eps}, min_samples={min_s})</h3>"))
    hasil = pd.DataFrame({"indeks_asli": idx, "cluster": labels})
    for c in NUMERIC_COLS:
        hasil[c] = sub[c].values
    display(hasil)
    for c in sorted(np.unique(labels)):
        anggota = [str(i) for i, l in zip(idx, labels) if l == c]
        print(f"Cluster {c} ({len(anggota)} data): {', '.join(anggota)}")
else:
    print("Metode tidak valid. Gunakan '1', '2', atau '3'.")"""),

    new_markdown_cell("""Selesai. Jalankan ulang sel Pengaturan dan Perhitungan jika ingin mengganti metode, indeks baris, atau jumlah cluster.""")
]

with open("program_clustering.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook ditulis: program_clustering.ipynb")
