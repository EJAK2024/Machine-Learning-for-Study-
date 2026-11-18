import pandas as pd
import numpy as np

INPUT = "daily_fitbit_sema_df_unprocessed.csv"
OUTPUT_DAILY = "daily_fitbit_sema_df_clean.csv"
OUTPUT_USER = "user_fitbit_sema_df_clean.csv"

# Kolom aktivitas inti yang tidak boleh kosong
COL_ACTIVITY = ["steps", "sedentary_minutes", "calories", "distance", "bpm"]
# Kolom tidur yang tidak boleh kosong
COL_SLEEP = ["minutesAsleep", "sleep_efficiency", "sleep_duration"]

df = pd.read_csv(INPUT)
print(f"Awal: {len(df)} baris, {df['id'].nunique()} user unik")

# 1. Konversi satuan waktu tidur: ms -> menit
df["sleep_duration_min"] = df["sleep_duration"] / 60000.0

# 2. Hapus baris dengan NaN pada kolom aktivitas inti dan kolom tidur
subset = COL_ACTIVITY + COL_SLEEP
df = df.dropna(subset=subset)
print(f"Setelah dropna(subset={subset}): {len(df)} baris")

# 3. Filter anomali sensor (wear-time validation)
df = df[(df["steps"] != 0) & (df["sedentary_minutes"] < 1440)]
df = df[df["minutesAsleep"] >= 60]
print(f"Setelah filter anomali: {len(df)} baris")

# Opsi A: Per hari (record harian) - sudah pada tingkat harian, simpan langsung
df_daily = df.reset_index(drop=True)
df_daily.to_csv(OUTPUT_DAILY, index=False)
print(f"Disimpan (per hari): {OUTPUT_DAILY} -> {len(df_daily)} baris")

# Opsi B: Per pengguna (profil per id) - rata-rata seluruh fitur numerik per id
num_cols = df_daily.select_dtypes(include=[np.number]).columns.tolist()
df_user = df_daily.groupby("id")[num_cols].mean().reset_index()
# Kembalikan kolom kategorikal dominant (mode) per id
cat_cols = ["age", "gender", "bmi", "step_goal_label"]
for c in cat_cols:
    if c in df_daily.columns:
        df_user[c] = df_daily.groupby("id")[c].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan).values
df_user = df_user.reset_index(drop=True)
df_user.to_csv(OUTPUT_USER, index=False)
print(f"Disimpan (per user): {OUTPUT_USER} -> {len(df_user)} baris")

print("\nSelesai.")
