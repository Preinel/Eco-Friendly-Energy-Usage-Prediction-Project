import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/Preinel/Eco-Friendly-Energy-Usage-Prediction-Project/refs/heads/main/smart_home_energy_consumption_large.csv")
print("Dataset Loaded Successfully\n")
print(df.head())

df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df["Hour"] = df["DateTime"].dt.hour
df["Day"] = df["DateTime"].dt.day
df["Month"] = df["DateTime"].dt.month
df["Weekday"] = df["DateTime"].dt.weekday
df["Is_Weekend"] = df["Weekday"].apply(lambda x: 1 if x >= 5 else 0)

df.drop(columns=["Date", "Time", "DateTime"], inplace=True)

print("\nAfter Feature Engineering:")
print(df.head())

#distribution graph
plt.figure()
sns.histplot(df["Energy Consumption (kWh)"], kde=True)
plt.title("Energy Consumption Distribution")
plt.show()

#season boxplot
plt.figure()
sns.boxplot(x="Season", y="Energy Consumption (kWh)", data=df)
plt.title("Energy Consumption by Season")
plt.show()

#hour lineplot graph
plt.figure()
sns.lineplot(x="Hour", y="Energy Consumption (kWh)", data=df)
plt.title("Energy Consumption by Hour")
plt.show()

#correlation heatmap
plt.figure()
sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nModel Selection:")
print("- Linear Regression")
print("- Decision Tree Regressor")
print("- Random Forest Regressor")
print("- Gradient Boosting Regressor")

df_encoded = pd.get_dummies(df, columns=["Appliance Type", "Season"], drop_first=True)

X = df_encoded.drop(columns=["Energy Consumption (kWh)"])
y = df_encoded["Energy Consumption (kWh)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

#linear regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#decision tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

#random forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

#gradient boosting
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

#metrics table
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
    "R2 Score": [
        r2_score(y_test, y_pred),
        r2_score(y_test, dt_pred),
        r2_score(y_test, rf_pred),
        r2_score(y_test, gb_pred)
    ],
    "MAE": [
        mean_absolute_error(y_test, y_pred),
        mean_absolute_error(y_test, dt_pred),
        mean_absolute_error(y_test, rf_pred),
        mean_absolute_error(y_test, gb_pred)
    ]
})

print("\nModel Comparison:")
print(results)

#actual vs predicted
plt.figure()

plt.scatter(y_test, y_pred, alpha=0.5, label="Linear Regression")
plt.scatter(y_test, dt_pred, alpha=0.5, label="Decision Tree")
plt.scatter(y_test, rf_pred, alpha=0.5, label="Random Forest")
plt.scatter(y_test, gb_pred, alpha=0.5, label="Gradient Boosting")

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")

plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted - Model Comparison")
plt.legend()
plt.show()

#r2 Comparison
plt.figure()
plt.bar(results["Model"], results["R2 Score"])
plt.xticks(rotation=45)
plt.title("R2 Score Comparison")
plt.ylabel("R2 Score")
plt.show()

#MAE comparison graph
plt.figure()
plt.bar(results["Model"], results["MAE"])
plt.xticks(rotation=45)
plt.title("MAE Comparison")
plt.ylabel("MAE")
plt.show()
