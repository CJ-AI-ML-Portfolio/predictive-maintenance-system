import unittest
import tensorflow as tf
from src.model_training import build_model, train_model
from src.data_preprocessing import load_and_preprocess_data


class TestModelTraining(unittest.TestCase):
    def test_model_building(self):
        """Test if model building returns a Sequential model."""
        X_train, _, _, _, _ = load_and_preprocess_data("data/raw/machine failure.csv")
        model = build_model(X_train.shape[1])
        self.assertIsInstance(
            model, tf.keras.Sequential, "Model should be a Sequential instance."
        )

    def test_model_training(self):
        """Test if model training completes without errors."""
        X_train, _, y_train, _, _ = load_and_preprocess_data(
            "data/raw/machine failure.csv"
        )
        try:
            train_model(X_train, y_train)
        except Exception as e:
            self.fail(f"Model training raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
