import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)

    # Handle missing values if any
    data.fillna(method="ffill", inplace=True)

    # Assume 'failure' is the target column and other columns are features
    X = data.drop(
        columns="failure"
    )  # Replace 'failure' with the actual target column if different
    y = data["failure"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    # Load data from 'machine failure.csv'
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(
        "data/raw/machine failure.csv"
    )
    print("Data preprocessing complete. Ready for model training.")
