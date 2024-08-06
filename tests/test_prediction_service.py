import unittest
import numpy as np
from src.prediction_service import predict_failure
from src.data_preprocessing import load_and_preprocess_data


class TestPredictionService(unittest.TestCase):
    def test_prediction_output(self):
        """Test if prediction returns a binary result."""
        _, _, _, _, scaler = load_and_preprocess_data("data/raw/machine failure.csv")
        new_data = np.random.normal(0, 1, 10)  # Replace with realistic test data
        prediction = predict_failure(new_data, scaler)
        self.assertIn(prediction, [0, 1], "Prediction should be either 0 or 1.")


if __name__ == "__main__":
    unittest.main()
