from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
import torch.nn as nn


# --- TensorFlow/Keras Models ---
def create_ffnn_model_tf(input_shape):
    model = Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=input_shape),
            layers.Dense(32, activation="relu"),
            layers.Dense(input_shape[0]),  # Output 23
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def create_lstm_model_tf(input_shape):
    model = Sequential(
        [
            layers.LSTM(50, activation="relu", input_shape=input_shape),
            layers.Dense(input_shape[0]),  # Fix: Output 23, not 1
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def create_cnn_model_tf(input_shape):
    model = Sequential(
        [
            layers.Reshape((input_shape[0], 1), input_shape=input_shape),
            layers.Conv1D(32, 3, activation="relu"),
            layers.Flatten(),
            layers.Dense(input_shape[0]),  # Output 23
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def create_transformer_model_tf(
    input_shape, ff_dim=4, mlp_units=None, mlp_dropout=0.4, dropout=0.25
):
    if mlp_units is None:
        mlp_units = [128]
    inputs = layers.Input(shape=input_shape)  # (None, 23)
    x = layers.Dense(64, activation="relu")(inputs)  # (None, 23, 64)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(ff_dim, activation="relu")(x)  # (None, 23, 4)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(input_shape[0])(x)  # (None, 23)
    model = Model(inputs, x)
    model.compile(optimizer="adam", loss="mse")
    return model


# --- PyTorch Models ---
class FFNNPyTorch(nn.Module):
    def __init__(self, input_size):
        super(FFNNPyTorch, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, input_size)  # Output 23
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LSTMPyTorch(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(LSTMPyTorch, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 23)  # Output 23
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        out = self.relu(out[:, -1, :])  # (batch, hidden_size)
        out = self.fc(out)  # (batch, 23)
        return out


class CNNPyTorch(nn.Module):
    def __init__(self, input_size):
        super(CNNPyTorch, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * input_size, 23)  # Output 23

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, time_steps)
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class TransformerPyTorch(nn.Module):
    def __init__(
        self,
        input_size,
        embed_dim=24,
        num_heads=4,
        ff_dim=4,
        num_layers=4,
        dropout=0.25,
    ):
        super(TransformerPyTorch, self).__init__()
        self.input_proj = nn.Linear(input_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(embed_dim, 23)  # Output 23
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # (batch, 1, embed_dim)
        x = self.transformer(x)
        x = x.squeeze(1)  # (batch, embed_dim)
        x = self.fc(x)  # (batch, 23)
        return x
