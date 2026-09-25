import pandas as pd

from config import (
    TRAIN_FILE,
    TEST_FILE,
    STORE_FILE,
    TRAIN_PROCESSED_FILE,
    TEST_PROCESSED_FILE,
    PROCESSED_DATA_DIR
)


def load_data():
    train = pd.read_csv(TRAIN_FILE, low_memory=False)
    test = pd.read_csv(TEST_FILE, low_memory=False)
    store = pd.read_csv(STORE_FILE)

    return train, test, store


def add_features(df, store):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    df = df.merge(store, on="Store", how="left")

    return df


def add_lag_features(train):
    train = train.sort_values(["Store", "Date"]).copy()

    train["Sales_Lag_1"] = train.groupby("Store")["Sales"].shift(1)
    train["Sales_Lag_7"] = train.groupby("Store")["Sales"].shift(7)
    train["Sales_Lag_14"] = train.groupby("Store")["Sales"].shift(14)

    train["Rolling_Mean_7"] = (
        train.groupby("Store")["Sales"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )

    train["Rolling_Mean_14"] = (
        train.groupby("Store")["Sales"]
        .transform(lambda x: x.shift(1).rolling(14).mean())
    )

    return train


def main():
    print("Starting feature engineering...")

    train, test, store = load_data()

    train = add_features(train, store)
    test = add_features(test, store)

    train = add_lag_features(train)

    train = train.dropna(
        subset=[
            "Sales_Lag_1",
            "Sales_Lag_7",
            "Sales_Lag_14",
            "Rolling_Mean_7",
            "Rolling_Mean_14"
        ]
    )

    test = test.sort_values(["Store", "Date"]).copy()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(TRAIN_PROCESSED_FILE, index=False)
    test.to_csv(TEST_PROCESSED_FILE, index=False)

    print("\nFeature engineering completed.")
    print(f"Processed train shape: {train.shape}")
    print(f"Processed test shape: {test.shape}")

    print("\nFeatures created:")
    print([
        "Year",
        "Month",
        "Day",
        "WeekOfYear",
        "IsWeekend",
        "Sales_Lag_1",
        "Sales_Lag_7",
        "Sales_Lag_14",
        "Rolling_Mean_7",
        "Rolling_Mean_14"
    ])

    print("\nProcessed files saved successfully.")


if __name__ == "__main__":
    main()