from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from data_preprocessing import load_and_preprocess_data


def build_model(input_shape):
    # Define the model architecture
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),  # Binary classification
        ]
    )
    return model


def train_model(X_train, y_train):
    model = build_model(X_train.shape[1])

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(
        X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1
    )

    # Save the trained model
    model.save("models/predictive_maintenance_model.h5")
    print("Model training complete and saved.")


if __name__ == "__main__":
    # Load and preprocess data from 'machine failure.csv'
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(
        "data/raw/machine failure.csv"
    )
    train_model(X_train, y_train)
