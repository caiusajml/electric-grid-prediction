import unittest
import numpy as np
from src.data_generator import generate_grid_data
from src.models import create_ffnn_model, create_lstm_model, create_cnn_model, create_transformer_model


class TestModels(unittest.TestCase):
    def test_data_generation(self):
        reshaped_load, _ = generate_grid_data(n_samples=10, time_steps=24)
        self.assertEqual(reshaped_load.shape, (10, 24))

    def test_model_creation(self):
        input_shape = (23,)
        ffnn = create_ffnn_model(input_shape)
        lstm = create_lstm_model((23, 1))
        cnn = create_cnn_model(input_shape)
        transformer = create_transformer_model(input_shape)

        self.assertIsNotNone(ffnn)
        self.assertIsNotNone(lstm)
        self.assertIsNotNone(cnn)
        self.assertIsNotNone(transformer)


if __name__ == '__main__':
    unittest.main()