import numpy as np
import tensorflow as tf
from data_preprocessing import load_and_preprocess_data


def load_model():
    # Load the saved model
    model = tf.keras.models.load_model("models/predictive_maintenance_model.h5")
    return model


def predict_failure(input_data, scaler):
    model = load_model()
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled).flatten()
    return (prediction > 0.5).astype(int)


if __name__ == "__main__":
    # Load and preprocess data to get the scaler
    _, _, _, _, scaler = load_and_preprocess_data("data/raw/machine failure.csv")

    # Example new data point (replace with actual sensor data from 'submission.csv')
    new_data = np.random.normal(0, 1, 10)  # Replace with actual data
    result = predict_failure(new_data, scaler)
    print("Predicted Failure:", result)
