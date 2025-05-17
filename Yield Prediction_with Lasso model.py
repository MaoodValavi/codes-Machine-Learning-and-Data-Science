import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# === Load Excel File ===
file_path = r'C:\Users\masud\Personallaptob12Jan2025\Freelancer\Parscoders\6_omran\final.xlsx'
df = pd.read_excel(file_path)

# === Define Target Columns ===   این ستونها همان خروجی های 
target_cols = ['y1', 'y2', 'y3', 'y4', 'y5']

# === Drop Identifier Columns (except keep 'Milgerd') ===
drop_cols = [col for col in df.columns if 'id' in col.lower() and col not in ['Milgerd']]
df = df.drop(columns=drop_cols)

# === Save ID for Sorting if Exists === چون می خواهیم ذر فایل اکسل نهایی کل داده ها را بر اساس آی دی سیو کنیم و مرتب کیم
id_col = None
for col in df.columns:
    if 'id' in col.lower():
        id_col = col
        break

# === Separate Features and Targets === 
X = df.drop(columns=target_cols)
y = df[target_cols]

# === Convert All Columns Except 'Milgerd' to Numeric === Diagonally Reinforced? column=its Milegerd Column

non_numeric_cols = ['Milgerd']
numeric_cols = [col for col in X.columns if col not in non_numeric_cols]
X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors='coerce')

# === Impute Missing Values in Numeric Columns ===  It fills Nan Values in feautre columns with median od each column so this two columns just get 0 and 1
X_imputer = SimpleImputer(strategy='median')
X[numeric_cols] = X_imputer.fit_transform(X[numeric_cols])

# === Encode 'Milgerd' Using One-Hot Encoding === the data type of this column is non numeric so we convert it here to numeric with dummy method to two columns Milegerd_Yes and Milegerd_No
X = pd.get_dummies(X, columns=non_numeric_cols, dummy_na=True)

# === Impute Missing y Using Lasso and Features === we use Lasso method.Explain
print("\n🔧 Imputing missing target values using features (Lasso Regression)...")
y_imputed = y.copy()
for col in target_cols:
    y_col = y[col].copy()
    mask_notna = y_col.notna()
    mask_na = y_col.isna()

    if mask_na.sum() == 0:
        print(f"✅ No missing values in {col}. Skipping.")
        continue

    model = Lasso(alpha=0.01, max_iter=10000, random_state=42)
    model.fit(X[mask_notna], y_col[mask_notna])
    y_pred = model.predict(X[mask_na])
    y_imputed.loc[mask_na, col] = y_pred
    print(f"🔄 {col}: filled {mask_na.sum()} missing values.")

# === Overwrite original DataFrame with imputed target values ===is used for generation of final Excel file
for col in target_cols:
    df[col] = y_imputed[col]

# === Cross-Validation Setup === cross validatation is used to check the how valid is our model. And then we later save our best performance
kf = KFold(n_splits=4, shuffle=True, random_state=42)
combined_results = df.copy()
combined_results = combined_results.drop(columns=target_cols)
combined_results['Set'] = ''  # Will store "Train" or "Test"

# === Process Each Target Separately ===
for col in target_cols:
    print(f"\n🔍 Processing target: {col}")

    best_r2 = -np.inf # just define it because we have to first define a variable
    best_preds = None
    best_set_labels = None
    best_model = None

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1): 
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] #X is the dataset used for cross validation which each time changs in 4 fold validation
        y_train, y_test = y_imputed[col].iloc[train_idx], y_imputed[col].iloc[test_idx]

        model = Lasso(alpha=0.01, max_iter=10000, random_state=42) #explain
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        r2_test = r2_score(y_test, y_test_pred) 
        # each time calculate r2 on the test and then we select the model that shows best performance on test

        if r2_test > best_r2:
            best_r2 = r2_test
            best_model = model

            # Combine predictions
            preds = pd.Series(index=X.index, dtype=float)
            set_labels = pd.Series(index=X.index, dtype=object)
            preds.iloc[train_idx] = y_train_pred
            preds.iloc[test_idx] = y_test_pred
            set_labels.iloc[train_idx] = "Train"
            set_labels.iloc[test_idx] = "Test"

            best_preds = preds
            best_set_labels = set_labels

    # Store predictions and actuals save all column
    combined_results[f'{col}_actual'] = y_imputed[col]
    combined_results[f'{col}_predicted'] = best_preds
    combined_results['Set'] = best_set_labels

    # === Print Lasso Equation ===
    coef_series = pd.Series(best_model.coef_, index=X.columns)
    intercept = best_model.intercept_
    equation = " + ".join([f"({coef:.4f}*{name})" for name, coef in coef_series.items() if coef != 0])
    print(f"\n📌 Lasso Equation for {col}:\n{col} = {intercept:.4f} + {equation}")

# === Optional: Sort by ID if exists ===
if id_col and id_col in df.columns:
    combined_results = pd.concat([df[[id_col]], combined_results], axis=1)
    combined_results = combined_results.sort_values(by=id_col)

# === Save Combined Results ===
output_combined = r'C:\Users\masud\Personallaptob12Jan2025\Freelancer\Parscoders\6_omran\combined_lasso_filled_by_features.xlsx'
combined_results.to_excel(output_combined, index=False)

print("\n📁 Combined results saved:")
print(f"   - File → {output_combined}")
