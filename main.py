import pandas as pd

# File Excel đầu vào
input_file = "vocab.xlsx"

# File CSV đầu ra
output_file = "output.csv"

# Đọc 2 cột đầu tiên
df = pd.read_excel(input_file, usecols=[0, 1])

# Xuất sang CSV
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Đã chuyển xong: {output_file}")
print(df.head())