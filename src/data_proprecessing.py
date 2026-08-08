import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


# Melakukan load data csv
dataset_insurance_claims = pd.read_csv("/home/nvoinxv/Documents/Classification_Predict_Accurance_Model/Data/Insurance claims data.csv")

# Menampilkan sample data dan mengecek tipe data
print("Beberapa sample data dari asuransi", dataset_insurance_claims.head(10))
print("Tipe data dari setiap kolom:")
print(dataset_insurance_claims.dtypes)

# Pemisahan dari kolom target dan features
target = "claim_status"

features_drop = {
    "region_density",
    "vehicle_age",
    "turning_radius",
    "airbags",
    "ncap_rating",
    "gross_weight",
    "policy_id",
    "is_esc",
    "is_tpms",
    "is_parking_camera",
    "rear_brakes_type",
    "transmission_type",
    "is_rear_window_wiper",
    "is_rear_window_washer",
    "is_rear_window_defogger",
    "is_power_door_locks",
    "is_central_locking",
    "is_power_steering",
    "is_ecw"
}

X = dataset_insurance_claims.drop(
    columns=[target, *features_drop]
)

y = dataset_insurance_claims[target]

# Mengecek korelasi pada data numerik
numeric_cols = dataset_insurance_claims.select_dtypes(
    include=["int64", "float64"]
).columns

target_corr = (
    dataset_insurance_claims[numeric_cols]
    .corr()["claim_status"]
    .drop("claim_status")
    .sort_values()
)

plt.figure(figsize=(10, 6))

target_corr.plot(kind="barh")

plt.xlabel("Correlation with Claim Status")
plt.title("Numerical Features vs Claim Status")
plt.tight_layout()
plt.show()

# Mengecek korelasi pada data string
categorical_cols = X.select_dtypes(include=["object"]).columns

for col in categorical_cols:

    table = pd.crosstab(
        X[col],
        y
    )

    chi2, p_value, dof, expected = chi2_contingency(table)

    print(f"{col}")
    print(f"Chi-square : {chi2:.4f}")
    print(f"p-value    : {p_value:.6f}")
    print("-" * 40)

# Mengecek sample data pada features
print("Sample data pada X", X.head(5))

# Melakukan preprocessing pada data features terutama string
categorical_features = X.select_dtypes(include=["object"]).columns

encoder = OneHotEncoder(
    handle_unknown="ignore"
)

X_encoded = encoder.fit_transform(
    X[categorical_features]
)

X_encoded_df = pd.DataFrame(
    X_encoded.toarray(),
    columns=encoder.get_feature_names_out(categorical_features),
    index=X.index
)

print(X_encoded_df.head())

encoder = LabelEncoder()

y_encode = pd.Series(
    encoder.fit_transform(y),
    index=y.index,
    name=y.name
)

print("Hasil label encoding pada target:", y_encode.head(5))

# Membuat plot untuk mengecek distribusi pada target agar seimbang
counts = y_encode.value_counts().sort_index()
labels = counts.index.astype(str)  # Label = nilai unik y_encode

# Sesuaikan explode dengan jumlah kategori sebenarnya
n_categories = len(counts)
explode = [0.05] + [0] * (n_categories - 1)  # Otomatis sesuai panjang

plt.figure(figsize=(5, 5))

plt.pie(
    counts,              # Data jumlah per kategori
    labels=labels,       # Label kategori
    autopct='%1.1f%%',
    startangle=90,
    explode=explode
)

plt.title('Distribusi Claim Status', fontsize=16)
plt.show()