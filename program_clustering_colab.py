# Program Clustering untuk Google Colab
# Salinan dari program_clustering.py, dibersihkan dari input() interaktif
# agar bisa dijalankan cell-by-cell di Google Colab / Jupyter Notebook.
# 5 metode: K-Means, K-Medoids/PAM, AHC (Average Linkage/UPGMA), DBSCAN, GMM

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


# ========================== PENGATURAN (ubah di sini) ==========================
CSV_PATH = "lifesnaps_cleaned_for_clustering.csv"  # di Colab: unggah file ini ke sesi Colab

NUMERIC_COLS = [
    "steps", "sedentary_minutes", "lightly_active_minutes",
    "moderately_active_minutes", "very_active_minutes",
    "calories", "minutesAsleep", "minutesAwake", "sleep_efficiency"
]

# Pilih metode: "1"=K-Means, "2"=K-Medoids, "3"=AHC, "4"=DBSCAN, "5"=GMM
METODE = "1"

# Indeks baris yang akan diproses. None = semua data. Contoh: [0, 1, 2, 5]
PILIH_INDEKS = None

# Jumlah cluster / komponen (digunakan untuk K-Means, K-Medoids, AHC, GMM)
K = 3

# Parameter DBSCAN
DBSCAN_EPS = 2.2
DBSCAN_MIN_SAMPLES = 18   # 2 x jumlah fitur (9 fitur)

# ========================== FUNGSI BANTU ==========================

def jarak_euclid(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2)))


def matriks_jarak(X):
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = jarak_euclid(X[i], X[j])
            D[j, i] = D[i, j]
    return D


# ---------- AHC Average Linkage (UPGMA) ----------
def ahc_average_linkage(X):
    n = X.shape[0]
    D = matriks_jarak(X)
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
                ds = [D[i][j] for i in members[c1] for j in members[c2]]
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
        if own_mask.any():
            a = float(np.mean([np.linalg.norm(X[i] - X[j]) for j in np.where(own_mask)[0]]))
        else:
            a = 0.0
        b = np.inf
        for c in uniq:
            if c == own:
                continue
            cm = labels == c
            dist = float(np.mean([np.linalg.norm(X[i] - X[j]) for j in np.where(cm)[0]]))
            if dist < b:
                b = dist
        s.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(s))


# ---------- K-Medoids (PAM sederhana) ----------
def k_medoids(X, k, max_iter=100, random_state=42):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    D = matriks_jarak(X)

    medoid_idx = rng.choice(n, size=k, replace=False)
    for _ in range(max_iter):
        labels = np.argmin(D[:, medoid_idx], axis=1)

        new_medoid = medoid_idx.copy()
        improved = False
        for m in range(k):
            non_medoid = [i for i in range(n) if i not in medoid_idx]
            best_cost = np.sum(D[labels == m][:, medoid_idx[m]]) if (labels == m).any() else np.inf
            best_candidate = medoid_idx[m]
            for cand in non_medoid:
                tmp = medoid_idx.copy()
                tmp[m] = cand
                tmp_labels = np.argmin(D[:, tmp], axis=1)
                cost = 0.0
                for cl in range(k):
                    members_cl = np.where(tmp_labels == cl)[0]
                    if len(members_cl) > 0:
                        cost += np.sum(D[members_cl][:, tmp[cl]])
                if cost < best_cost:
                    best_cost = cost
                    best_candidate = cand
                    improved = True
            new_medoid[m] = best_candidate

        medoid_idx = new_medoid
        if not improved:
            break

    labels = np.argmin(D[:, medoid_idx], axis=1)
    return labels, medoid_idx


# ========================== EKSEKUSI UTAMA ==========================

def jalankan():
    # Muat data
    df = pd.read_csv(CSV_PATH)
    print(f"Data dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    print("Kolom numerik:", NUMERIC_COLS)

    # Pilih baris
    if PILIH_INDEKS is None:
        idx = list(df.index)
    else:
        idx = [i for i in PILIH_INDEKS if i in df.index]

    sub = df.loc[idx].copy()
    X = sub[NUMERIC_COLS].to_numpy(dtype=float)
    print(f"Jumlah data yang diproses: {len(idx)}")

    # Validasi k
    k = max(2, min(K, len(idx)))

    # ================= METODE 1: K-Means =================
    if METODE == "1":
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        print("\n================ HASIL K-MEANS ================")
        print(f"Jumlah data : {len(idx)}")
        print(f"Jumlah cluster (k) : {k}")
        print(f"Inersia : {km.inertia_:.4f}")

        hasil = pd.DataFrame({"indeks_asli": idx, "cluster": labels})
        for c in NUMERIC_COLS:
            hasil[c] = sub[c].values
        print(hasil.to_string(index=False))

        print("\n-- Anggota Setiap Cluster --")
        for c in sorted(np.unique(labels)):
            anggota = [str(i) for i, l in zip(idx, labels) if l == c]
            print(f"Cluster {c} ({len(anggota)} data): {', '.join(anggota)}")

        sil = hitung_silhouette(X, labels)
        print(f"\nSilhouette score : {sil:.4f}")

    # ================= METODE 2: K-Medoids / PAM =================
    elif METODE == "2":
        labels, medoid_idx = k_medoids(X, k)
        print("\n================ HASIL K-MEDOIDS (PAM) ================")
        print(f"Jumlah data : {len(idx)}")
        print(f"Jumlah cluster (k) : {k}")
        print(f"Indeks medoid (pada data yang dipilih): {list(medoid_idx)}")

        hasil = pd.DataFrame({"indeks_asli": idx, "cluster": labels})
        for c in NUMERIC_COLS:
            hasil[c] = sub[c].values
        print(hasil.to_string(index=False))

        print("\n-- Anggota Setiap Cluster --")
        for c in sorted(np.unique(labels)):
            anggota = [str(i) for i, l in zip(idx, labels) if l == c]
            print(f"Cluster {c} ({len(anggota)} data): {', '.join(anggota)}")

        sil = hitung_silhouette(X, labels)
        print(f"\nSilhouette score : {sil:.4f}")

    # ================= METODE 3: AHC Average Linkage (UPGMA) =================
    elif METODE == "3":
        if len(idx) > 300:
            print("\nPERINGATAN: AHC dengan banyak data akan sangat lambat (O(n^3)).")
            print("Disarankan gunakan subset kecil atau pilih K-Means.")
        linkage, D = ahc_average_linkage(X)
        labels = potong_dendrogram(linkage, len(idx), k)

        print("\n================ HASIL AHC — AVERAGE LINKAGE (UPGMA) ================")
        print(f"Jumlah data : {len(idx)}")
        print(f"Jumlah cluster (k) : {k}")

        print("\n-- Matriks Jarak Euclidean --")
        np.set_printoptions(precision=3, suppress=True)
        print(np.round(D, 3))

        print("\n-- Linkage Matrix --")
        print("Format: [cluster1, cluster2, jarak, ukuran_cluster_baru]")
        for row in linkage:
            print(f"  {int(row[0]):>4}  {int(row[1]):>4}  {row[2]:>8.4f}  {int(row[3])}")

        hasil = pd.DataFrame({"indeks_asli": idx, "cluster": labels})
        for c in NUMERIC_COLS:
            hasil[c] = sub[c].values
        print(hasil.to_string(index=False))

        print("\n-- Anggota Setiap Cluster --")
        for c in sorted(np.unique(labels)):
            anggota = [str(i) for i, l in zip(idx, labels) if l == c]
            print(f"Cluster {c} ({len(anggota)} data): {', '.join(anggota)}")

        sil = hitung_silhouette(X, labels)
        print(f"\nSilhouette score : {sil:.4f}")

    # ================= METODE 4: DBSCAN =================
    elif METODE == "4":
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        labels = db.fit_predict(X)

        print("\n================ HASIL DBSCAN ================")
        print(f"Jumlah data : {len(idx)}")
        print(f"eps : {DBSCAN_EPS}")
        print(f"min_samples : {DBSCAN_MIN_SAMPLES}")

        hasil = pd.DataFrame({"indeks_asli": idx, "cluster": labels})
        for c in NUMERIC_COLS:
            hasil[c] = sub[c].values
        print(hasil.to_string(index=False))

        n_noise = int(np.sum(labels == -1))
        print(f"\nJumlah titik noise (cluster -1): {n_noise}")

        uniq = np.unique(labels)
        valid_clusters = len(uniq) - (1 if -1 in uniq else 0)
        if valid_clusters > 1:
            mask = labels != -1
            sil = hitung_silhouette(X[mask], labels[mask])
            print(f"Silhouette score (non-noise) : {sil:.4f}")
        else:
            print("Silhouette score tidak dapat dihitung (kurang dari 2 klaster non-noise).")

    # ================= METODE 5: GMM =================
    elif METODE == "5":
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
        labels = gmm.fit_predict(X)

        print("\n================ HASIL GMM ================")
        print(f"Jumlah data : {len(idx)}")
        print(f"Jumlah komponen : {k}")
        print(f"Log-likelihood : {gmm.score(X):.4f}")

        hasil = pd.DataFrame({"indeks_asli": idx, "cluster": labels})
        for c in NUMERIC_COLS:
            hasil[c] = sub[c].values
        print(hasil.to_string(index=False))

        print("\n-- Anggota Setiap Cluster --")
        for c in sorted(np.unique(labels)):
            anggota = [str(i) for i, l in zip(idx, labels) if l == c]
            print(f"Cluster {c} ({len(anggota)} data): {', '.join(anggota)}")

        sil = hitung_silhouette(X, labels)
        print(f"\nSilhouette score : {sil:.4f}")

    else:
        print("Metode tidak valid. Gunakan '1', '2', '3', '4', atau '5'.")

    print("\n=== SELESAI ===")


if __name__ == "__main__":
    jalankan()
