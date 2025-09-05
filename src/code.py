
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load Dataset
df = pd.read_csv(r"C:\Users\gvnar\OneDrive\Desktop\Placement\Projects\water tds\github\data\water_quality_dataset.csv", encoding='latin1')

# Quick look
print(df.head())
print(df.info())
print(df.describe())


# Exploratory Data Analysis (EDA)
# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Scatter plot: Conductivity vs TDS
sns.scatterplot(x="Conductivity (µS/cm)", y="TDS (mg/L)", data=df)
plt.title("Conductivity vs TDS")
plt.show()

#  Handle Missing Values
# (Synthetic dataset has none, but this is general)
df.fillna(df.mean(), inplace=True)


#  Feature Selection
X = df[["Temperature (°C)", "pH", "Turbidity (NTU)", "Conductivity (µS/cm)", "DO (mg/L)"]]
y = df["TDS (mg/L)"]

#  Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Feature Scaling (Optional for Linear Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Evaluation
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print("Linear Regression Performance:")
print(f"R²: {r2_lr:.3f}, RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}")

# Plot Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_lr, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual TDS")
plt.ylabel("Predicted TDS")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()


# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Random Forest can handle unscaled data
y_pred_rf = rf_model.predict(X_test)

# Evaluation
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("Random Forest Performance:")
print(f"R²: {r2_rf:.3f}, RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}")

# Plot Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual TDS")
plt.ylabel("Predicted TDS")
plt.title("Random Forest: Actual vs Predicted")
plt.show()

# Feature Importance (Random Forest)
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importance.sort_values(ascending=True).plot(kind='barh', figsize=(6,4))
plt.title("Random Forest Feature Importance")
plt.show()

