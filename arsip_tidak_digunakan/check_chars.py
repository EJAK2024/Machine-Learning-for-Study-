line = "AHC awalnya menggunakan linkage='ward', diubah menjadi linkage='average' (Average Linkage / UPGMA) sesuai permintaan."
for i, c in enumerate(line):
    if ord(c) >= 128:
        print(f"Index {i}: U+{ord(c):04X} {repr(c)}")
print("All chars checked")
