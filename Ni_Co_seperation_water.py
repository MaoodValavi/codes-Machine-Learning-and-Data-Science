import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Load data
df = pd.read_csv("fluorescence_data.csv")

# Initial inspection
print("First 5 rows of data:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Separate features and targets
target_cols = ['Ni_Concentration_mgl', 'Co_Concentration_mgl']  # change based on actual column names
exclude_cols = target_cols + ['Sample_Type']  # ignore text columns for now
feature_cols = [col for col in df.columns if col not in exclude_cols]

X = df[feature_cols]
y = df[target_cols]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Multivariate linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel Performance:")
print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# Visualize actual vs predicted
plt.figure(figsize=(12, 5))

for i, metal in enumerate(target_cols):
    plt.subplot(1, 2, i+1)
    sns.scatterplot(x=y_test[metal], y=y_pred[:, i])
    plt.xlabel(f"Actual {metal}")
    plt.ylabel(f"Predicted {metal}")
    plt.title(f"{metal} Prediction")

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Fluorescence spectrum visualization (example)
fluorescence_cols = [col for col in df.columns if "nm" in col or "intensity" in col.lower()]
if fluorescence_cols:
    sample_idx = 0  # choose first sample
    spectrum = df.loc[sample_idx, fluorescence_cols]
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(fluorescence_cols)), spectrum.values, marker='o')
    plt.title(f"Fluorescence Spectrum - Sample {sample_idx}")
    plt.xlabel("Wavelength Index")
    plt.ylabel("Fluorescence Intensity")
    plt.xticks(ticks=range(len(fluorescence_cols)), labels=fluorescence_cols, rotation=90)
    plt.tight_layout()
    plt.show()
else:
    print("No fluorescence columns found for spectrum visualization.")

