
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/Preinel/Eco-Friendly-Energy-Usage-Prediction-Project/refs/heads/main/smart_home_energy_consumption_large.csv")
print("Dataset Loaded Successfully\n")
print(df.head())
print("\nDataset Info:")
print(df.info())

df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])

df["Hour"] = df["DateTime"].dt.hour
df["Day"] = df["DateTime"].dt.day
df["Month"] = df["DateTime"].dt.month
df["Weekday"] = df["DateTime"].dt.weekday
df["Is_Weekend"] = df["Weekday"].apply(lambda x: 1 if x >= 5 else 0)

df.drop(columns=["Date", "Time", "DateTime"], inplace=True)

print("\nAfter Feature Engineering:")
print(df.head())

#distribution plot graph
plt.figure()
sns.histplot(df["Energy Consumption (kWh)"], kde=True)
plt.title("Energy Consumption Distribution")
plt.show()

# boxplot by season
plt.figure()
sns.boxplot(x="Season", y="Energy Consumption (kWh)", data=df)
plt.title("Energy Consumption by Season")
plt.show()

# lineplot graph by hour
plt.figure()
sns.lineplot(x="Hour", y="Energy Consumption (kWh)", data=df)
plt.title("Energy Consumption by Hour")
plt.show()

# heatmap
plt.figure()
sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
#regression models 
print("\nModel Selection:")
print("- Linear Regression")
print("- Decision Tree Regressor")
print("- Random Forest Regressor")
print("- Gradient Boosting Regressor")

df_encoded = pd.get_dummies(df, columns=["Appliance Type", "Season"], drop_first=True)

# Define Features and Target
X = df_encoded.drop(columns=["Energy Consumption (kWh)"])
y = df_encoded["Energy Consumption (kWh)"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Check accuracy
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Optimized R2:", r2_score(y_test, rf_pred))
print("Optimized MAE:", mean_absolute_error(y_test, rf_pred))
import matplotlib.pyplot as plt

plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("Actual vs Predicted")
plt.show()


