import pandas as pd

# 1. Memuat dataset LifeSnaps
df = pd.read_csv('daily_fitbit_sema_df_unprocessed.csv')

# 2. Memilih kolom identitas dan fitur aktivitas fisik + kualitas tidur
selected_cols = [
    'id', 'date',
    'steps', 'sedentary_minutes', 'lightly_active_minutes',
    'moderately_active_minutes', 'very_active_minutes', 'calories',
    'minutesAsleep', 'minutesAwake', 'sleep_efficiency'
]
df_clean = df[selected_cols].copy()

# 3. Menghapus baris dengan nilai kosong (missing values)
df_clean = df_clean.dropna()

# 4. Menghapus anomali perangkat tidak dipakai (steps = 0 atau sedenter penuh 1440 menit)
df_clean = df_clean[(df_clean['steps'] > 0) & (df_clean['sedentary_minutes'] < 1440)]
df_clean = df_clean[df_clean['minutesAsleep'] >= 60]

# 5. Memisahkan kolom fitur numerik untuk clustering
features = [
    'steps', 'sedentary_minutes', 'lightly_active_minutes',
    'moderately_active_minutes', 'very_active_minutes', 'calories',
    'minutesAsleep', 'minutesAwake', 'sleep_efficiency'
]
X = df_clean[features]

print(f"Jumlah data siap analisis: {X.shape[0]} baris, {X.shape[1]} variabel")
df_clean.to_csv('lifesnaps_cleaned_for_clustering.csv', index=False)