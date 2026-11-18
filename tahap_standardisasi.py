import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT = "lifesnaps_cleaned_for_clustering.csv"
OUTPUT = "lifesnaps_scaled.csv"

df = pd.read_csv(INPUT)
id_cols = ["id", "date"]
feature_cols = [c for c in df.columns if c not in id_cols]

X = df[feature_cols].copy()
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
Xs = pd.DataFrame(Xs, columns=feature_cols, index=df.index)

df_scaled = pd.concat([df[id_cols].reset_index(drop=True), Xs.reset_index(drop=True)], axis=1)
df_scaled.to_csv(OUTPUT, index=False)

print(f"Input : {INPUT}  {df.shape}")
print(f"Output: {OUTPUT}  {df_scaled.shape}")
print(f"Fitur distandardisasi ({len(feature_cols)}): {feature_cols}")
print("\nContoh 5 baris pertama (sudah Z-score):")
print(df_scaled.head())
print("\nStatistik ringkasan (harusnya mean ~0, std ~1):")
print(df_scaled[feature_cols].describe().loc[["mean", "std"]].round(4))
