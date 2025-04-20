import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
df = pd.read_csv("incomeStatementHistory_annually.csv")

# Select Relevant Columns
features = [
    "grossProfit", "operatingIncome", "costOfRevenue", "totalOperatingExpenses",
    "netIncome", "incomeBeforeTax", "netIncomeFromContinuingOps", "interestExpense"
]
target = "totalRevenue"

# Drop rows with missing values in features OR target
df = df[features + [target]].dropna()

# Apply Log Transformation to Target Variable
df[target] = np.log1p(df[target].where(df[target] > 0, 1e-10))

# Split Data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)

# Save the model and scaler
pickle.dump(rf, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model and Scaler saved!")
