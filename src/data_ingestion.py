import pandas as pd

from config import TRAIN_FILE, TEST_FILE, STORE_FILE


def load_data():
    train = pd.read_csv(TRAIN_FILE, low_memory=False)
    test = pd.read_csv(TEST_FILE, low_memory=False)
    store = pd.read_csv(STORE_FILE)

    return train, test, store


if __name__ == "__main__":
    train, test, store = load_data()

    print("Data ingestion successful")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Store shape: {store.shape}")