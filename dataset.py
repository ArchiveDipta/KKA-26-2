import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. LOAD DATA
df = pd.read_csv('data_praktikum_analisis_data (1) - data_praktikum_analisis_data (1).csv')

# Rapikan nama kolom
df.columns = df.columns.str.strip().str.lower()

print("Kolom:", df.columns)
print(df.head())


# 2. DETEKSI KOLOM OTOMATIS
col = {}

for c in df.columns:
    if "product" in c:
        col["product"] = c
    elif "customer" in c:
        col["customer"] = c
    elif "date" in c:
        col["date"] = c
    elif "order" in c and "id" in c:
        col["order_id"] = c
    elif "price" in c:
        col["price"] = c
    elif "sales" in c:
        col["sales"] = c
    elif "quantity" in c or "qty" in c:
        col["qty"] = c
    elif "ad" in c or "budget" in c:
        col["ads"] = c

print("Mapping:", col)

# 3. DATA CLEANING
if "date" in col:
    df[col["date"]] = pd.to_datetime(df[col["date"]], errors='coerce')

df = df.dropna()

# 4. RFM ANALYSIS
if all(k in col for k in ["customer", "date", "sales", "order_id"]):
    snapshot_date = df[col["date"]].max() + dt.timedelta(days=1)

    rfm = df.groupby(col["customer"]).agg({
        col["date"]: lambda x: (snapshot_date - x.max()).days,
        col["order_id"]: "count",
        col["sales"]: "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # aman dari error qcut
    rfm['R'] = pd.qcut(rfm['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
    rfm['F'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['M'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])

    rfm['RFM'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

    print("\nTop Customer:")
    print(rfm.sort_values("Monetary", ascending=False).head())

# 5. TOP PRODUK
if "product" in col and "sales" in col:
    top_produk = df.groupby(col["product"])[col["sales"]].sum().sort_values(ascending=False)

    print("\nTop Produk:")
    print(top_produk.head())

    plt.figure()
    top_produk.head(10).plot(kind='bar')
    plt.title("Top 10 Produk Terlaris")
    plt.xticks(rotation=45)
    plt.show()

# 6. TREN BULANAN
if "date" in col and "sales" in col:
    df["month"] = df[col["date"]].dt.to_period("M").astype(str)

    monthly = df.groupby("month")[col["sales"]].sum()

    plt.figure()
    plt.plot(monthly.index, monthly.values, marker='o')
    plt.title("Tren Penjualan Bulanan")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

# 7. KORELASI
corr_cols = []
for k in ["sales", "ads", "qty", "price"]:
    if k in col:
        corr_cols.append(col[k])

if len(corr_cols) >= 2:
    corr = df[corr_cols].corr()

    plt.figure()
    sns.heatmap(corr, annot=True)
    plt.title("Korelasi")
    plt.show()

# 8. REGRESI LINEAR
if "ads" in col and "sales" in col:
    X = df[[col["ads"]]]
    y = df[col["sales"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nKoefisien Iklan:", model.coef_[0])
    print("R2 Score:", model.score(X_test, y_test))

# 9. INSIGHT SINGKAT
print("\nInsight:")
print("- Identifikasi produk terlaris")
print("- Lihat tren penjualan bulanan")
print("- Analisis pelanggan terbaik (RFM)")
print("- Evaluasi pengaruh iklan terhadap penjualan")