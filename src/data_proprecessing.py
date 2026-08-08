import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from scipy.stats import chi2_contingency


class InsuranceClaimPreprocessor:
    def __init__(
        self,
        data_path,
        target="claim_status",
        corr_threshold=0.05,
        iqr_factor=1.5,
        features_drop=None
    ):
        self.data_path = data_path
        self.target = target
        self.corr_threshold = corr_threshold
        self.iqr_factor = iqr_factor
        self.features_drop = features_drop or {
            "region_density", "vehicle_age", "turning_radius", "airbags",
            "ncap_rating", "gross_weight", "policy_id", "is_esc", "is_tpms",
            "is_parking_camera", "rear_brakes_type", "transmission_type",
            "is_rear_window_wiper", "is_rear_window_washer",
            "is_rear_window_defogger", "is_power_door_locks",
            "is_central_locking", "is_power_steering", "is_ecw"
        }

        self.df = None
        self.X = None
        self.y = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.le = LabelEncoder()

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        return self

    def drop_features(self):
        cols_to_drop = [self.target] + list(self.features_drop)
        cols_to_drop = [c for c in cols_to_drop if c in self.df.columns]
        self.X = self.df.drop(columns=cols_to_drop)
        self.y = self.df[self.target]
        return self

    def plot_outliers(self, save_path=None):
        self.numeric_cols = self.X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        n_cols = 3
        n_rows = int(np.ceil(len(self.numeric_cols) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
        axes = axes.flatten() if len(self.numeric_cols) > 1 else [axes]

        boxprops = dict(facecolor="#FFD700", edgecolor="black", linewidth=1.5)
        medianprops = dict(color="black", linewidth=2)
        whiskerprops = dict(color="black", linewidth=1.5)
        capprops = dict(color="black", linewidth=1.5)
        flierprops = dict(marker="o", markerfacecolor="black", markersize=5, alpha=0.6)

        for i, col in enumerate(self.numeric_cols):
            ax = axes[i]
            sns.boxplot(
                x=self.X[col], ax=ax, width=0.5,
                boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops
            )
            ax.set_title(col, fontsize=11, fontweight="bold", color="black")
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=9, colors="black")
            ax.set_facecolor("white")
            ax.grid(False)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle("Boxplot Outlier Detection", fontsize=16, fontweight="bold", color="black")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.show()
        return self

    def remove_outliers(self):
        mask = np.ones(len(self.X), dtype=bool)
        for col in self.numeric_cols:
            arr = self.X[col].to_numpy()
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lower = q1 - self.iqr_factor * iqr
            upper = q3 + self.iqr_factor * iqr
            mask &= (arr >= lower) & (arr <= upper)
        self.X = self.X[mask].reset_index(drop=True)
        self.y = self.y[mask].reset_index(drop=True)
        return self

    def drop_low_correlation(self):
        temp = self.X.copy()
        temp[self.target] = self.y
        numeric_cols = temp.select_dtypes(include=["int64", "float64"]).columns
        corr = temp[numeric_cols].corr()[self.target].drop(self.target).abs().sort_values()
        low_corr = corr[corr < self.corr_threshold].index.tolist()
        if low_corr:
            self.X = self.X.drop(columns=low_corr)
            print(f"Dropped low correlation: {low_corr}")
        return self

    def chi_square_test(self):
        cat_cols = self.X.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            table = pd.crosstab(self.X[col], self.y)
            chi2, p, _, _ = chi2_contingency(table)
            print(f"{col}: chi2={chi2:.4f}, p={p:.6f}")
        return self

    def scale_numeric(self):
        self.numeric_cols = self.X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if self.numeric_cols:
            self.X[self.numeric_cols] = self.scaler.fit_transform(self.X[self.numeric_cols])
        return self

    def encode_categorical(self):
        self.categorical_cols = self.X.select_dtypes(include=["object"]).columns.tolist()
        if not self.categorical_cols:
            return self
        encoded = self.ohe.fit_transform(self.X[self.categorical_cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.ohe.get_feature_names_out(self.categorical_cols),
            index=self.X.index
        )
        self.X = pd.concat([self.X.drop(columns=self.categorical_cols), encoded_df], axis=1)
        return self

    def encode_target(self):
        self.y = pd.Series(
            self.le.fit_transform(self.y),
            index=self.y.index,
            name=self.y.name
        )
        return self

    def plot_target_distribution(self):
        counts = self.y.value_counts().sort_index()
        explode = [0.05] + [0] * (len(counts) - 1)
        plt.figure(figsize=(5, 5))
        plt.pie(counts, labels=counts.index.astype(str), autopct="%1.1f%%",
                startangle=90, explode=explode)
        plt.title("Distribusi Claim Status", fontsize=16)
        plt.show()
        return self

    def get_data(self):
        return self.X, self.y

    def run(self):
        return (
            self.load_data()
            .drop_features()
            .plot_outliers()
            .remove_outliers()
            .drop_low_correlation()
            .chi_square_test()
            .scale_numeric()
            .encode_categorical()
            .encode_target()
            .plot_target_distribution()
            .get_data()
        )


if __name__ == "__main__":
    preprocessor = InsuranceClaimPreprocessor(
        data_path="/home/nvoinxv/Documents/Classification_Predict_Accurance_Model/Data/Insurance claims data.csv"
    )
    X_processed, y_encoded = preprocessor.run()
    print(f"\nFinal shape: X={X_processed.shape}, y={y_encoded.shape}")
    print(X_processed.head())
