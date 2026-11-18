import numpy as np
import pandas as pd

CSV_PATH = "lifesnaps_cleaned_for_clustering.csv"
NUMERIC_COLS = ["steps", "sedentary_minutes", "lightly_active_minutes",
                "moderately_active_minutes", "very_active_minutes",
                "calories", "minutesAsleep", "minutesAwake", "sleep_efficiency"]


def muat_data(path):
    return pd.read_csv(path)


def tampilkan_tabel(df, limit=15):
    print("\n=== DATA (indeks pada kolom paling kiri) ===")
    print(df.head(limit).to_string())
    print(f"... total {len(df)} baris.")
    print("Kolom numerik:", NUMERIC_COLS)


def pilih_metode():
    print("\n=== PILIH METODE CLUSTERING ===")
    print("1. K-Means")
    print("2. K-Medoids")
    print("3. AHC - Average Linkage (UPGMA)")
    print("4. DBSCAN")
    print("5. GMM (Gaussian Mixture Model)")
    while True:
        m = input("Masukkan pilihan metode (1/2/3/4/5): ").strip()
        if m in ("1", "2", "3", "4", "5"):
            return m
        print("Pilihan tidak valid. Masukkan 1, 2, 3, 4, atau 5.")


def pilih_data(df):
    print("\n=== PILIH DATA YANG AKAN DIITI ===")
    print("Masukkan indeks baris, pisahkan koma. Contoh: 0, 1, 2, 5")
    print("Ketik 'all' untuk menggunakan semua data.")
    while True:
        raw = input("Pilihan Anda: ").strip()
        if raw.lower() == "all":
            return list(df.index)
        try:
            idx = [int(x.strip()) for x in raw.split(",") if x.strip()]
            valid = [i for i in idx if i in df.index]
            invalid = [i for i in idx if i not in df.index]
            if invalid:
                print(f"Indeks tidak ditemukan dan diabaikan: {invalid}")
            if not valid:
                print("Tidak ada indeks valid. Coba lagi.")
                continue
            return valid
        except ValueError:
            print("Format salah. Masukkan angka pisahkan koma.")


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


# ---------------- AHC Average Linkage (UPGMA) ----------------
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


# ---------------- K-Medoids (PAM sederhana) ----------------
def k_medoids(X, k, max_iter=100, random_state=42):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    D = matriks_jarak(X)

    medoid_idx = rng.choice(n, size=k, replace=False)
    for _ in range(max_iter):
        # assign tiap titik ke medoid terdekat
        labels = np.argmin(D[:, medoid_idx], axis=1)

        # coba swap tiap medoid dengan titik non-medoid
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


# ---------------- Tampilkan hasil ----------------
def cetak_hasil_umum(nama_metode, idx, labels, sub, k=None, extra_info="", tampilkan_header=True):
    if tampilkan_header:
        print(f"\n================ HASIL {nama_metode.upper()} ================")
        print(f"Jumlah data yang dihitung : {len(idx)}")
    if k:
        print(f"Jumlah cluster            : {k}")
    if extra_info:
        print(extra_info)

    print("\n-- Hasil Cluster per Baris --")
    hasil = pd.DataFrame({"indeks_asli": idx, "cluster": labels})
    for c in NUMERIC_COLS:
        hasil[c] = sub[c].values
    print(hasil.to_string(index=False))

    print("\n-- Anggota Setiap Cluster --")
    for c in sorted(np.unique(labels)):
        anggota = [str(i) for i, l in zip(idx, labels) if l == c]
        print(f"Cluster {c} ({len(anggota)} data): {', '.join(anggota)}")

    X = sub[NUMERIC_COLS].to_numpy(dtype=float)
    sil = hitung_silhouette(X, labels)
    print(f"\nSilhouette score          : {sil:.4f}")


def jalankan_kmeans(sub, idx, k):
    from sklearn.cluster import KMeans
    X = sub[NUMERIC_COLS].to_numpy(dtype=float)
    k = max(2, min(k, len(idx)))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    cetak_hasil_umum("K-Means", idx, labels, sub, k=k,
                     extra_info=f"Inersia : {km.inertia_:.4f}")


def jalankan_kmedoids(sub, idx, k):
    X = sub[NUMERIC_COLS].to_numpy(dtype=float)
    k = max(2, min(k, len(idx)))
    labels, medoid_idx = k_medoids(X, k)
    cetak_hasil_umum("K-Medoids", idx, labels, sub, k=k,
                     extra_info=f"Indeks medoid (pada data yang dipilih): {list(medoid_idx)}")


def jalankan_ahc(sub, idx, k):
    X = sub[NUMERIC_COLS].to_numpy(dtype=float)
    k = max(2, min(k, len(idx)))
    linkage, D = ahc_average_linkage(X)
    labels = potong_dendrogram(linkage, len(idx), k)

    print("\n================ HASIL AHC — AVERAGE LINKAGE (UPGMA) ================")
    print(f"Jumlah data yang dihitung : {len(idx)}")
    print(f"Jumlah cluster (k)        : {k}")

    print("\n-- Matriks Jarak Euclidean --")
    np.set_printoptions(precision=3, suppress=True)
    print(np.round(D, 3))

    print("\n-- Linkage Matrix --")
    print("Format: [cluster1, cluster2, jarak, ukuran_cluster_baru]")
    for row in linkage:
        print(f"  {int(row[0]):>4}  {int(row[1]):>4}  {row[2]:>8.4f}  {int(row[3])}")

    cetak_hasil_umum("AHC Average Linkage", idx, labels, sub, k=k, tampilkan_header=False)


def jalankan_dbscan(sub, idx, eps, min_s):
    from sklearn.cluster import DBSCAN
    X = sub[NUMERIC_COLS].to_numpy(dtype=float)
    db = DBSCAN(eps=eps, min_samples=min_s)
    labels = db.fit_predict(X)

    print("\n================ HASIL DBSCAN ================")
    print(f"Jumlah data yang dihitung : {len(idx)}")
    print(f"eps                       : {eps}")
    print(f"min_samples               : {min_s}")

    cetak_hasil_umum("DBSCAN", idx, labels, sub)

    n_noise = int(np.sum(labels == -1))
    print(f"\nJumlah titik noise (cluster -1): {n_noise}")


def jalankan_gmm(sub, idx, k):
    from sklearn.mixture import GaussianMixture
    X = sub[NUMERIC_COLS].to_numpy(dtype=float)
    k = max(2, min(k, len(idx)))
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
    labels = gmm.fit_predict(X)

    print("\n================ HASIL GMM ================")
    print(f"Jumlah data yang dihitung : {len(idx)}")
    print(f"Jumlah komponen           : {k}")
    print(f"Log-likelihood            : {gmm.score(X):.4f}")

    cetak_hasil_umum("GMM", idx, labels, sub, k=k)


def main():
    print("=== PROGRAM CLUSTERING lifesnaps_cleaned_for_clustering.csv ===")
    df = muat_data(CSV_PATH)
    tampilkan_tabel(df)

    metode = pilih_metode()
    idx = pilih_data(df)
    sub = df.loc[idx].copy()

    print(f"\nMenyiapkan {len(idx)} baris data untuk metode {metode} ...")

    if len(idx) > 300 and metode == "3":
        konf = input(f"Peringatan: AHC dengan {len(idx)} baris akan lambat (O(n^3)). Lanjutkan? (y/n): ").strip().lower()
        if konf != "y":
            print("Dibatalkan oleh user.")
            return

    if metode == "1":
        k_input = input("Masukkan jumlah cluster k (default 3): ").strip()
        k = int(k_input) if k_input else 3
        jalankan_kmeans(sub, idx, k)
    elif metode == "2":
        k_input = input("Masukkan jumlah cluster k (default 3): ").strip()
        k = int(k_input) if k_input else 3
        jalankan_kmedoids(sub, idx, k)
    elif metode == "3":
        k_input = input("Masukkan jumlah cluster k (default 3): ").strip()
        k = int(k_input) if k_input else 3
        jalankan_ahc(sub, idx, k)
    elif metode == "4":
        eps_input = input("Masukkan eps (default 0.5): ").strip()
        eps = float(eps_input) if eps_input else 0.5
        ms_input = input("Masukkan min_samples (default 5): ").strip()
        min_s = int(ms_input) if ms_input else 5
        jalankan_dbscan(sub, idx, eps, min_s)
    elif metode == "5":
        k_input = input("Masukkan jumlah komponen k (default 3): ").strip()
        k = int(k_input) if k_input else 3
        jalankan_gmm(sub, idx, k)

    print("\n=== SELESAI ===")


if __name__ == "__main__":
    main()
