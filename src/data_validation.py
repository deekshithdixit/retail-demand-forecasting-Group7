import pandas as pd

from config import TRAIN_FILE, TEST_FILE, STORE_FILE


def validate_required_columns(df, required_columns, dataset_name):
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        print(f"❌ {dataset_name}: Missing columns: {missing_columns}")
        return False

    print(f"✓ {dataset_name}: Required columns present")
    return True


def validate_duplicates(df, dataset_name):
    duplicates = df.duplicated().sum()

    if duplicates == 0:
        print(f"✓ {dataset_name}: No duplicate rows")
        return True

    print(f"❌ {dataset_name}: {duplicates} duplicate rows found")
    return False


def validate_sales(train):
    negative_sales = (train["Sales"] < 0).sum()

    if negative_sales == 0:
        print("✓ Train: No negative Sales values")
        return True

    print(f"❌ Train: {negative_sales} negative Sales values found")
    return False


def validate_store_ids(train, test, store):
    train_ids = set(train["Store"])
    test_ids = set(test["Store"])
    store_ids = set(store["Store"])

    invalid_train = train_ids - store_ids
    invalid_test = test_ids - store_ids

    valid = True

    if invalid_train:
        print(f"❌ Train: {len(invalid_train)} invalid Store IDs")
        valid = False
    else:
        print("✓ Train: All Store IDs exist in store data")

    if invalid_test:
        print(f"❌ Test: {len(invalid_test)} invalid Store IDs")
        valid = False
    else:
        print("✓ Test: All Store IDs exist in store data")

    return valid


def validate_dates(train, test):
    train_dates = pd.to_datetime(train["Date"], errors="coerce")
    test_dates = pd.to_datetime(test["Date"], errors="coerce")

    invalid_train = train_dates.isna().sum()
    invalid_test = test_dates.isna().sum()

    valid = True

    if invalid_train == 0:
        print("✓ Train: All Date values are valid")
    else:
        print(f"❌ Train: {invalid_train} invalid Date values")
        valid = False

    if invalid_test == 0:
        print("✓ Test: All Date values are valid")
    else:
        print(f"❌ Test: {invalid_test} invalid Date values")
        valid = False

    return valid


def validate_data(train, test, store):
    print("========================================")
    print("        RETAIL DATA VALIDATION")
    print("========================================\n")

    validation_results = []

    train_columns = [
        "Store",
        "DayOfWeek",
        "Date",
        "Sales",
        "Customers",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday"
    ]

    test_columns = [
        "Id",
        "Store",
        "DayOfWeek",
        "Date",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday"
    ]

    store_columns = [
        "Store",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval"
    ]

    print("1. REQUIRED COLUMN CHECKS")
    validation_results.append(
        validate_required_columns(train, train_columns, "Train")
    )
    validation_results.append(
        validate_required_columns(test, test_columns, "Test")
    )
    validation_results.append(
        validate_required_columns(store, store_columns, "Store")
    )

    print("\n2. DUPLICATE CHECKS")
    validation_results.append(validate_duplicates(train, "Train"))
    validation_results.append(validate_duplicates(test, "Test"))
    validation_results.append(validate_duplicates(store, "Store"))

    print("\n3. TARGET VALIDATION")
    validation_results.append(validate_sales(train))

    print("\n4. STORE ID VALIDATION")
    validation_results.append(validate_store_ids(train, test, store))

    print("\n5. DATE VALIDATION")
    validation_results.append(validate_dates(train, test))

    print("\n6. MISSING VALUE SUMMARY")
    print(f"Train missing values: {train.isnull().sum().sum()}")
    print(f"Test missing values: {test.isnull().sum().sum()}")
    print(f"Store missing values: {store.isnull().sum().sum()}")

    print("\n========================================")

    if all(validation_results):
        print("DATA VALIDATION PASSED")
        print("========================================")
        return True

    print("DATA VALIDATION FAILED")
    print("========================================")
    return False


if __name__ == "__main__":
    train = pd.read_csv(TRAIN_FILE, low_memory=False)
    test = pd.read_csv(TEST_FILE, low_memory=False)
    store = pd.read_csv(STORE_FILE)

    validate_data(train, test, store)