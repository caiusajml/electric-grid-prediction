from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_generator import generate_grid_data
from src.models import (
    create_ffnn_model_tf,
    create_lstm_model_tf,
    create_cnn_model_tf,
    create_transformer_model_tf,
    FFNNPyTorch,
    LSTMPyTorch,
    CNNPyTorch,
    TransformerPyTorch,
)
import torch
import torch.nn as nn
import torch.optim as optim


def train_models(framework="tensorflow"):
    n_samples, time_steps = 500, 24
    reshaped_load, _ = generate_grid_data(n_samples, time_steps)

    scaler = StandardScaler()
    scaled_load = scaler.fit_transform(reshaped_load)

    x_scaled_load = scaled_load[:, :-1]
    y_scaled_load = scaled_load[:, 1:]
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled_load, y_scaled_load, test_size=0.2, shuffle=False
    )

    models = []

    if framework.lower() == "tensorflow":
        input_shape = x_train.shape[1:]
        print(f"TensorFlow input_shape: {input_shape}")
        for i, create_func in enumerate(
            [
                create_ffnn_model_tf,
                create_lstm_model_tf,
                create_cnn_model_tf,
                create_transformer_model_tf,
            ]
        ):
            if i == 1:  # LSTM
                model = create_func((x_train.shape[1], 1))
            else:
                model = create_func(input_shape)
            print(f"Model {i} ({model.__class__.__name__}):")
            model.summary()
            models.append(model)
        x_train_lstm = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

        for i, model in enumerate(models):
            print(f"Training Model {i} ({model.__class__.__name__})...")
            if i == 1:  # LSTM
                model.fit(x_train_lstm, y_train, epochs=50, batch_size=32, verbose=0)
            else:
                model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)

    elif framework.lower() == "pytorch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_train_torch = torch.tensor(x_train, dtype=torch.float32).to(
            device
        )  # [400, 23]
        y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(
            device
        )  # [400, 23]
        x_train_lstm = x_train_torch.unsqueeze(-1)  # [400, 23, 1]

        models = [
            FFNNPyTorch(x_train.shape[1]).to(device),
            LSTMPyTorch(1).to(device),
            CNNPyTorch(x_train.shape[1]).to(device),
            TransformerPyTorch(x_train.shape[1], embed_dim=24, num_heads=4).to(device),
        ]

        for model in models:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            for epoch in range(50):  # 50 epochs
                optimizer.zero_grad()
                if isinstance(model, LSTMPyTorch):
                    output = model(x_train_lstm)
                else:
                    output = model(x_train_torch)
                if output is None:
                    raise ValueError(
                        f"Model {model.__class__.__name__} returned None in forward pass"
                    )
                try:
                    print(
                        f"Model: {model.__class__.__name__}, Epoch: {epoch}, Output shape: {output.size()}"
                    )
                    loss = criterion(output, y_train_torch)
                    loss.backward()
                    optimizer.step()
                except AttributeError as e:
                    raise ValueError(
                        f"Model {model.__class__.__name__} output issue: {str(e)}"
                    )

    return x_train, x_test, y_train, y_test, scaler, models


if __name__ == "__main__":
    train_models("tensorflow")  # Default to TensorFlow
