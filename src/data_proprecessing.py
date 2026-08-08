import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


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
    "policy_id"
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

