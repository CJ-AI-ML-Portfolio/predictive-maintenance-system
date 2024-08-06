import unittest
import numpy as np
from src.data_preprocessing import load_and_preprocess_data


class TestDataPreprocessing(unittest.TestCase):
    def test_data_loading(self):
        """Test if data loading returns expected shapes."""
        X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(
            "data/raw/machine failure.csv"
        )
        self.assertGreater(len(X_train), 0, "Training data should not be empty.")
        self.assertGreater(len(X_test), 0, "Test data should not be empty.")

    def test_data_scaling(self):
        """Test if scaling is applied correctly."""
        X_train, X_test, _, _, scaler = load_and_preprocess_data(
            "data/raw/machine failure.csv"
        )
        self.assertAlmostEqual(
            np.mean(X_train),
            0,
            delta=0.1,
            msg="Training data should be centered around 0 after scaling.",
        )
        self.assertAlmostEqual(
            np.std(X_train),
            1,
            delta=0.1,
            msg="Training data should have unit variance after scaling.",
        )


if __name__ == "__main__":
    unittest.main()
