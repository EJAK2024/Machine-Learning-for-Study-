from fpdf import FPDF

def sanitize(text):
    return "".join(c if ord(c) < 128 else "?" for c in str(text))

class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 60, 120)
        self.cell(0, 10, sanitize("Dokumentasi Tahap 3 s.d. Tahap 10 - Clustering LifeSnaps"), new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, sanitize("Proyek PEMSIN - Analisis Gaya Hidup Sedenter"), new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(5)

    def chapter_title(self, num, title):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(220, 230, 245)
        self.set_text_color(30, 60, 120)
        self.cell(0, 8, sanitize("BAB " + str(num) + ": " + title), new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(3)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 90, 150)
        self.cell(0, 7, sanitize(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text, size=9):
        self.set_font("Helvetica", "", size)
        self.set_text_color(0, 0, 0)
        # Use write() instead of multi_cell to avoid line breaking issues
        self.write(5, sanitize(text))
        self.ln(4)

    def bullet(self, text, size=9):
        self.set_font("Helvetica", "", size)
        self.set_text_color(0, 0, 0)
        self.cell(5, 5, "-", new_x="RIGHT", new_y="TOP")
        self.write(5, sanitize(text))
        self.ln(4)

    def table(self, headers, rows, col_widths=None, size=8):
        self.set_font("Helvetica", "B", size)
        self.set_text_color(255, 255, 255)
        self.set_fill_color(70, 110, 160)
        if col_widths is None:
            col_widths = [self.w / len(headers)] * len(headers)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, sanitize(h), border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", size)
        self.set_text_color(0, 0, 0)
        self.set_fill_color(255, 255, 255)
        for row in rows:
            for i, cell in enumerate(row):
                fill = i % 2 == 0
                self.cell(col_widths[i], 6, sanitize(cell), border=1, fill=fill)
            self.ln()
        self.ln(4)

pdf = PDF()
pdf.add_page()

# BAB 1
pdf.chapter_title(1, "Ringkasan Proyek")
pdf.body_text("Dokumentasi ini mencakup semua program yang telah dibuat untuk proses clustering pada dataset LifeSnaps, mulai dari Tahap 3 (Pembersihan Data) hingga Tahap 10 (Interpretasi Klaster dan Visualisasi). Semua file disimpan di folder D:\\SMSTR 5\\PEMSIN\\UAS\\csv_rais_anonymized")
pdf.body_text("Dataset yang digunakan: daily_fitbit_sema_df_unprocessed.csv (7.410 baris awal)")

# BAB 2
pdf.chapter_title(2, "File Program yang Dibuat")
pdf.body_text("Berikut daftar file program baru yang telah dibuat:")

files = [
    ("tahap3_pembersihan_data.py", "Tahap 3: Pembersihan data (konversi sleep_duration ms ke menit, dropna, filter anomali sensor, agregasi per hari dan per user). Output: daily_fitbit_sema_df_clean.csv dan user_fitbit_sema_df_clean.csv."),
    ("tahap4_clustering.py", "Tahap 4: StandardScaler dan eksekusi 5 algoritma clustering (KMeans, Agglomerative, DBSCAN, GMM, Spectral). Output: lifesnaps_clustered.csv dan clustering_comparison.csv."),
    ("tahap_standardisasi.py", "Tahap 6: Standardisasi Z-Score (StandardScaler) pada lifesnaps_cleaned_for_clustering.csv ke lifesnaps_scaled.csv."),
    ("tahap7_parameter_optimal.py", "Tahap 7: Penentuan parameter optimal (Elbow, Silhouette, Dendrogram, BIC/AIC, k-distance graph). Output: grafik di folder tahap7_output dan parameter_rekomendasi.csv."),
    ("tahap8_eksekusi_clustering.py", "Tahap 8: Eksekusi 5 model clustering dan simpan label ke lifesnaps_clustered.csv."),
    ("tahap9_evaluasi.py", "Tahap 9: Evaluasi performa dengan Silhouette, Davies-Bouldin, Calinski-Harabaz, persentase noise. Output: tahap9_perbandingan_clustering.csv."),
    ("tahap10_interpretasi.py", "Tahap 10: Cluster profiling, label risiko sedenter, PCA scatter, radar chart, bar plot. Output di folder tahap10_output."),
    ("tahap6_10_gabungan.py", "Gabungan lengkap Tahap 6 s.d. 10 (StandardScaler, 5 model, evaluasi, profiling). File utama untuk dijalankan. Output: lifesnaps_clustered_final.csv dan tahap9_perbandingan_clustering.csv."),
]
for fname, desc in files:
    pdf.section_title(fname)
    pdf.body_text(desc)

# BAB 3
pdf.chapter_title(3, "Perubahan pada File Utama tahap6_10_gabungan.py")
pdf.body_text("File gabungan ini mengalami beberapa revisi:")
pdf.bullet("Awalnya menggunakan sklearn_extra.cluster.KMedoids, tetapi gagal build di environment. Diganti dengan implementasi K-Medoids PAM bawaan class KMedoidsPAM.")
pdf.bullet("AHC awalnya menggunakan linkage=ward, diubah menjadi linkage=average (Average Linkage UPGMA) sesuai permintaan.")
pdf.bullet("DBSCAN eps awalnya 12.579 (outlier maks dari k-distance, menghasilkan 1 klaster), dikoreksi menjadi eps=2.2 (di dekat knee bulk persentil-90) sehingga DBSCAN menghasilkan klaster yang bermakna.")
pdf.bullet("K diatur menjadi K=3 untuk K-Means, K-Medoids, AHC, GMM; DBSCAN menentukan jumlah klaster otomatis.")

# BAB 4
pdf.chapter_title(4, "Hasil Evaluasi 5 Model Tahap 9")
pdf.body_text("Berdasarkan data lifesnaps_scaled.csv (3.529 baris x 9 fitur, Z-Score), K=3, DBSCAN eps=2.2, min_samples=18:")

headers = ["Metode", "Klaster", "Silhouette (tinggi)", "Davies-Bouldin (rendah)", "Calinski-Harabasz (tinggi)", "Noise (%)"]
rows = [
    ["K-Means", "3", "0.1684", "1.6258", "915.28", "0.0"],
    ["K-Medoids", "3", "0.2681", "1.2916", "823.84", "0.0"],
    ["AHC Average Linkage UPGMA", "3", "0.6985", "0.5815", "53.37", "0.0"],
    ["DBSCAN", "2", "0.5017", "0.6289", "102.15", "3.2"],
    ["GMM", "3", "0.0454", "4.6556", "184.40", "0.0"],
]
pdf.table(headers, rows)
pdf.body_text("Kesimpulan: Berdasarkan Silhouette Score dan Davies-Bouldin Index, AHC Average Linkage terbaik (0.6985 / 0.5815). DBSCAN juga unggul tetapi memiliki 3.2 persen noise dan klaster tidak seimbang.")

# BAB 5
pdf.chapter_title(5, "Profil Klaster K-Means Skala Asli")
pdf.body_text("Hasil grouping K-Means (K=3) pada data asli (belum scaling):")
headers2 = ["Klaster", "n", "steps", "sedentary_minutes", "sleep_efficiency"]
rows2 = [
    ["0", "1.478", "6.883", "674.65", "93.01"],
    ["1", "1.146", "5.891", "854.96", "95.18"],
    ["2", "905", "15.438", "607.41", "93.26"],
]
pdf.table(headers2, rows2)

pdf.bullet("Klaster 0 dan 1: Langkah rendah, duduk lama, cenderung risiko sedenter atau transisi.")
pdf.bullet("Klaster 2: Langkah tinggi (15.438), duduk lebih sedikit, gaya hidup aktif dan tidur sehat.")

pdf.body_text("Distribusi kategori risiko sedenter (berdasarkan aturan rules-based):")
headers3 = ["Kategori", "Jumlah"]
rows3 = [
    ["Transisi Deteksi Dini Sedenter", "3.210"],
    ["Aktif dan Tidur Sehat", "309"],
    ["Risiko Sedenter Tinggi", "10"],
]
pdf.table(headers3, rows3)

# BAB 6
pdf.chapter_title(6, "Cara Menjalankan Program")
pdf.body_text("1. Buka PowerShell di folder D:\\SMSTR 5\\PEMSIN\\UAS\\csv_rais_anonymized")
pdf.body_text("2. Pastikan file lifesnaps_cleaned_for_clustering.csv sudah ada (hasil Tahap 3).")
pdf.body_text("3. Jalankan perintah: python tahap6_10_gabungan.py")
pdf.body_text("4. Output yang dihasilkan:")
pdf.bullet("lifesnaps_clustered_final.csv - data asli ditambah 5 kolom label klaster dan kategori risiko sedenter")
pdf.bullet("tahap9_perbandingan_clustering.csv - tabel evaluasi performa 5 metode")
pdf.bullet("folder tahap7_output - grafik parameter optimal")
pdf.bullet("folder tahap10_output - visualisasi PCA, radar chart, bar plot")

pdf.output("Dokumentasi_Tahap3_sampai_Tahap10_Clustering.pdf")
print("PDF berhasil dibuat: Dokumentasi_Tahap3_sampai_Tahap10_Clustering.pdf")
