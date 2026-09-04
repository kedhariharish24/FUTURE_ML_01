from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# FUTURE INTERNS - MACHINE LEARNING
# TASK 1: SALES & DEMAND FORECASTING FOR BUSINESSES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Walmart dataset
DATA_FILE = BASE_DIR / "Walmart.csv"

# Folder for results
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n========== LOADING DATA ==========")

df = pd.read_csv(DATA_FILE)

print("Original rows:", len(df))
print("Columns:")
print(df.columns.tolist())


# ============================================================
# 2. DATA CLEANING
# ============================================================

print("\n========== DATA CLEANING ==========")

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

numeric_columns = [
    "Weekly_Sales",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

print("\nRows after cleaning:", len(df))


# ============================================================
# 3. SELECT STORE 1
# ============================================================

print("\n========== STORE SELECTION ==========")

store_number = 1

df = df[df["Store"] == store_number].copy()

df = df.sort_values("Date")

print("Selected Store:", store_number)

print("Rows:", len(df))

print(
    "Date range:",
    df["Date"].min().date(),
    "to",
    df["Date"].max().date()
)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\n========== FEATURE ENGINEERING ==========")

# Time features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
df["Day"] = df["Date"].dt.day

# Time index
df["Time_Index"] = np.arange(len(df))

# Lag features
df["Lag_1"] = df["Weekly_Sales"].shift(1)
df["Lag_4"] = df["Weekly_Sales"].shift(4)
df["Lag_12"] = df["Weekly_Sales"].shift(12)

# Rolling averages
df["Rolling_4"] = (
    df["Weekly_Sales"]
    .shift(1)
    .rolling(4)
    .mean()
)

df["Rolling_12"] = (
    df["Weekly_Sales"]
    .shift(1)
    .rolling(12)
    .mean()
)

# Remove rows created by lag features
df = df.dropna().reset_index(drop=True)

print("Time and lag features created successfully.")


# ============================================================
# 5. FEATURES AND TARGET
# ============================================================

features = [
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Year",
    "Month",
    "Quarter",
    "Week",
    "Day",
    "Time_Index",
    "Lag_1",
    "Lag_4",
    "Lag_12",
    "Rolling_4",
    "Rolling_12"
]

target = "Weekly_Sales"


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\n========== TRAIN / TEST SPLIT ==========")

# Last 12 weeks for testing
test_size = 12

train = df.iloc[:-test_size].copy()

test = df.iloc[-test_size:].copy()

X_train = train[features]

y_train = train[target]

X_test = test[features]

y_test = test[target]

print("Training rows:", len(train))

print("Testing rows :", len(test))


# ============================================================
# 7. TRAIN RANDOM FOREST MODEL
# ============================================================

print("\n========== MODEL TRAINING ==========")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print(
    "Random Forest model trained successfully."
)


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

print("\n========== MODEL PERFORMANCE ==========")

predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print(f"MAE  : {mae:,.2f}")

print(f"RMSE : {rmse:,.2f}")

print(f"R2   : {r2:.4f}")


# Save metrics
metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2 Score"
    ],
    "Value": [
        mae,
        rmse,
        r2
    ]
})

metrics.to_csv(
    OUTPUT_DIR / "model_metrics.csv",
    index=False
)


# ============================================================
# 9. SAVE TEST PREDICTIONS
# ============================================================

test_predictions = test[
    [
        "Date",
        "Weekly_Sales",
        "Holiday_Flag"
    ]
].copy()

test_predictions[
    "Predicted_Sales"
] = predictions

test_predictions.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)


# ============================================================
# 10. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    test["Date"],
    y_test,
    label="Actual Sales"
)

plt.plot(
    test["Date"],
    predictions,
    label="Predicted Sales"
)

plt.title(
    "Actual vs Predicted Weekly Sales - Store 1"
)

plt.xlabel("Date")

plt.ylabel("Weekly Sales")

plt.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=150
)

plt.close()


# ============================================================
# 11. MONTHLY SALES TREND
# ============================================================

monthly_sales = (
    df.set_index("Date")[
        "Weekly_Sales"
    ]
    .resample("ME")
    .sum()
    .reset_index()
)

monthly_sales.to_csv(
    OUTPUT_DIR / "monthly_sales.csv",
    index=False
)

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    monthly_sales["Date"],
    monthly_sales["Weekly_Sales"]
)

plt.title(
    "Monthly Sales Trend - Store 1"
)

plt.xlabel("Date")

plt.ylabel(
    "Total Monthly Sales"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_sales_trend.png",
    dpi=150
)

plt.close()


# ============================================================
# 12. FUTURE 12-WEEK FORECAST
# ============================================================

print("\n========== FUTURE FORECAST ==========")

history = df.copy()

future_rows = []

last_date = history["Date"].max()


for i in range(1, 13):

    future_date = (
        last_date
        + pd.Timedelta(weeks=i)
    )

    sales_history = (
        history["Weekly_Sales"]
        .tolist()
    )

    lag_1 = sales_history[-1]

    lag_4 = sales_history[-4]

    lag_12 = sales_history[-12]

    rolling_4 = np.mean(
        sales_history[-4:]
    )

    rolling_12 = np.mean(
        sales_history[-12:]
    )

    future_row = {

        "Holiday_Flag": 0,

        "Temperature":
            history["Temperature"]
            .tail(8)
            .mean(),

        "Fuel_Price":
            history["Fuel_Price"]
            .tail(8)
            .mean(),

        "CPI":
            history["CPI"]
            .tail(8)
            .mean(),

        "Unemployment":
            history["Unemployment"]
            .tail(8)
            .mean(),

        "Year":
            future_date.year,

        "Month":
            future_date.month,

        "Quarter":
            future_date.quarter,

        "Week":
            int(
                future_date
                .isocalendar()
                .week
            ),

        "Day":
            future_date.day,

        "Time_Index":
            len(history),

        "Lag_1":
            lag_1,

        "Lag_4":
            lag_4,

        "Lag_12":
            lag_12,

        "Rolling_4":
            rolling_4,

        "Rolling_12":
            rolling_12
    }

    future_X = pd.DataFrame(
        [future_row]
    )[features]

    forecast = model.predict(
        future_X
    )[0]

    future_rows.append({

        "Date":
            future_date,

        "Forecast_Sales":
            forecast

    })

    new_history_row = future_row.copy()

    new_history_row["Date"] = (
        future_date
    )

    new_history_row[
        "Weekly_Sales"
    ] = forecast

    history = pd.concat(
        [
            history,

            pd.DataFrame(
                [new_history_row]
            )
        ],
        ignore_index=True
    )


future_forecast = pd.DataFrame(
    future_rows
)

print(
    future_forecast.to_string(
        index=False
    )
)

future_forecast.to_csv(
    OUTPUT_DIR /
    "future_12_week_forecast.csv",
    index=False
)


# ============================================================
# 13. FUTURE FORECAST GRAPH
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    future_forecast["Date"],
    future_forecast["Forecast_Sales"],
    marker="o"
)

plt.title(
    "Future 12-Week Sales Forecast - Store 1"
)

plt.xlabel("Date")

plt.ylabel(
    "Forecasted Weekly Sales"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "future_12_week_forecast.png",
    dpi=150
)

plt.close()


# ============================================================
# 14. BUSINESS INSIGHTS
# ============================================================

average_sales = (
    df["Weekly_Sales"].mean()
)

maximum_sales = (
    df["Weekly_Sales"].max()
)

minimum_sales = (
    df["Weekly_Sales"].min()
)

average_forecast = (
    future_forecast[
        "Forecast_Sales"
    ].mean()
)

maximum_forecast = (
    future_forecast[
        "Forecast_Sales"
    ].max()
)

minimum_forecast = (
    future_forecast[
        "Forecast_Sales"
    ].min()
)

holiday_sales = (
    df.loc[
        df["Holiday_Flag"] == 1,
        "Weekly_Sales"
    ].mean()
)

non_holiday_sales = (
    df.loc[
        df["Holiday_Flag"] == 0,
        "Weekly_Sales"
    ].mean()
)


insights = f"""

FUTURE INTERNS - TASK 1
SALES & DEMAND FORECASTING

BUSINESS INSIGHTS
=================

1. Historical average weekly sales:
   {average_sales:,.2f}

2. Highest historical weekly sales:
   {maximum_sales:,.2f}

3. Lowest historical weekly sales:
   {minimum_sales:,.2f}

4. Average forecasted weekly sales:
   {average_forecast:,.2f}

5. Highest forecasted weekly sales:
   {maximum_forecast:,.2f}

6. Lowest forecasted weekly sales:
   {minimum_forecast:,.2f}

7. Average holiday-week sales:
   {holiday_sales:,.2f}

8. Average non-holiday sales:
   {non_holiday_sales:,.2f}


BUSINESS RECOMMENDATIONS
========================

- Use the forecast for inventory planning.

- Prepare sufficient stock before periods
  with higher predicted sales.

- Monitor holiday periods because sales
  behavior may change.

- Use sales forecasts to support
  purchasing and staffing decisions.

- Retrain the model periodically when
  new sales data becomes available.

"""


with open(
    OUTPUT_DIR /
    "business_insights.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(insights)


# ============================================================
# 15. COMPLETION MESSAGE
# ============================================================

print("\n========================================")

print(
    "TASK 1 MODEL COMPLETED SUCCESSFULLY"
)

print("========================================")

print("\nOutput files saved in:")

print(OUTPUT_DIR)

print("\nCreated files:")

print("- model_metrics.csv")

print("- test_predictions.csv")

print("- actual_vs_predicted.png")

print("- monthly_sales.csv")

print("- monthly_sales_trend.png")

print("- future_12_week_forecast.csv")

print("- future_12_week_forecast.png")

print("- business_insights.txt")