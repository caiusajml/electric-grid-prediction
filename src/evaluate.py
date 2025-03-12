import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .train import train_models
import torch


def evaluate_models(framework="tensorflow"):
    x_train, x_test, y_train, y_test, scaler, models = train_models(framework)

    if framework.lower() == "tensorflow":
        ffnn_model, lstm_model, cnn_model, transformer_model = models
        ffnn_pred = ffnn_model.predict(x_test, verbose=0)
        lstm_pred = lstm_model.predict(
            x_test.reshape(x_test.shape[0], x_test.shape[1], 1), verbose=0
        )
        cnn_pred = cnn_model.predict(x_test, verbose=0)
        transformer_pred = transformer_model.predict(x_test, verbose=0)

    elif framework.lower() == "pytorch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_test_torch = torch.tensor(x_test, dtype=torch.float32).to(device)
        x_test_lstm = x_test_torch.unsqueeze(-1)

        ffnn_model, lstm_model, cnn_model, transformer_model = models
        with torch.no_grad():
            ffnn_pred = ffnn_model(x_test_torch).cpu().numpy()
            lstm_pred = lstm_model(x_test_lstm).cpu().numpy()
            cnn_pred = cnn_model(x_test_torch).cpu().numpy()
            transformer_pred = transformer_model(x_test_torch).cpu().numpy()

    metrics = {
        "FFNN": [
            mean_squared_error(y_test, ffnn_pred),
            mean_absolute_error(y_test, ffnn_pred),
        ],
        "LSTM": [
            mean_squared_error(y_test, lstm_pred),
            mean_absolute_error(y_test, lstm_pred),
        ],
        "CNN": [
            mean_squared_error(y_test, cnn_pred),
            mean_absolute_error(y_test, cnn_pred),
        ],
        "Transformer": [
            mean_squared_error(y_test, transformer_pred),
            mean_absolute_error(y_test, transformer_pred),
        ],
    }

    print("\nModel Performance:")
    for model_name, (mse, mae) in metrics.items():
        print(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}")

    plt.figure(figsize=(12, 6))
    sample_idx = 0
    plt.plot(y_test[sample_idx], label="Actual", linewidth=2)
    plt.plot(ffnn_pred[sample_idx], label="FFNN")
    plt.plot(lstm_pred[sample_idx], label="LSTM")
    plt.plot(cnn_pred[sample_idx], label="CNN")
    plt.plot(transformer_pred[sample_idx], label="Transformer")
    plt.title(f"Model Predictions vs Actual ({framework.capitalize()})")
    plt.xlabel("Hour")
    plt.ylabel("Scaled Load")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    evaluate_models("tensorflow")
