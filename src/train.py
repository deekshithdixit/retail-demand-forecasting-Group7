import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import TRAIN_PROCESSED_FILE, MODEL_FILE, MODEL_DIR


def main():
    print("Loading processed training data...")

    df = pd.read_csv(TRAIN_PROCESSED_FILE)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    feature_columns = [
        "Store",
        "DayOfWeek",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "Year",
        "Month",
        "Day",
        "WeekOfYear",
        "IsWeekend",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
        "Sales_Lag_1",
        "Sales_Lag_7",
        "Sales_Lag_14",
        "Rolling_Mean_7",
        "Rolling_Mean_14"
    ]

    target = "Sales"

    X = df[feature_columns]
    y = df[target]

    split_date = df["Date"].quantile(0.8)

    train_mask = df["Date"] <= split_date
    validation_mask = df["Date"] > split_date

    X_train = X[train_mask]
    X_val = X[validation_mask]

    y_train = y[train_mask]
    y_val = y[validation_mask]

    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Split date: {split_date.date()}")

    categorical_features = [
        "StateHoliday",
        "StoreType",
        "Assortment",
        "PromoInterval"
    ]

    numerical_features = [
        column for column in feature_columns
        if column not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    print("Preparing features...")

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    print(f"Processed training shape: {X_train_processed.shape}")

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    )

    print("Training model...")

    model.fit(X_train_processed, y_train)

    print("Model training completed.")

    predictions = model.predict(X_val_processed)

    mae = mean_absolute_error(y_val, predictions)
    rmse = mean_squared_error(y_val, predictions) ** 0.5

    print("\nModel Evaluation")
    print("----------------")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "features": feature_columns
        },
        MODEL_FILE
    )

    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()