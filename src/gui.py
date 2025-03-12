import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.train import train_models
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch


class GridPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Electric Grid Load Prediction")
        self.root.geometry("800x600")

        self.models = None
        self.X_test = None
        self.y_test = None
        self.scaler = None
        self.framework = None

        self.create_widgets()

    def create_widgets(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill="x")

        # Framework selection
        ttk.Label(control_frame, text="Framework:").pack(side="left", padx=5)
        self.framework_var = tk.StringVar(value="TensorFlow")
        ttk.Combobox(
            control_frame,
            textvariable=self.framework_var,
            values=["TensorFlow", "PyTorch"],
            state="readonly",
        ).pack(side="left", padx=5)

        ttk.Button(control_frame, text="Train Models", command=self.train_models).pack(
            side="left", padx=5
        )
        ttk.Button(
            control_frame, text="Evaluate & Show Metrics", command=self.show_metrics
        ).pack(side="left", padx=5)

        ttk.Label(control_frame, text="Sample Index:").pack(side="left", padx=5)
        self.sample_var = tk.StringVar(value="0")
        ttk.Entry(control_frame, textvariable=self.sample_var, width=5).pack(
            side="left", padx=5
        )
        ttk.Button(
            control_frame, text="Plot Predictions", command=self.plot_predictions
        ).pack(side="left", padx=5)

        self.metrics_text = tk.Text(self.root, height=10, width=80)
        self.metrics_text.pack(pady=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def train_models(self):
        try:
            self.framework = self.framework_var.get().lower()
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(
                tk.END, f"Training models with {self.framework.capitalize()}...\n"
            )
            self.root.update()

            X_train, X_test, y_train, y_test, scaler, models = train_models(
                self.framework
            )
            self.X_train, self.X_test, self.y_train, self.y_test = (
                X_train,
                X_test,
                y_train,
                y_test,
            )
            self.scaler = scaler
            self.models = models

            self.metrics_text.insert(tk.END, "Training complete!\n")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")

    def show_metrics(self):
        if not self.models:
            messagebox.showwarning("Warning", "Please train models first!")
            return

        ffnn_model, lstm_model, cnn_model, transformer_model = self.models
        X_test, y_test = self.X_test, self.y_test

        if self.framework == "tensorflow":
            ffnn_pred = ffnn_model.predict(X_test, verbose=0)
            lstm_pred = lstm_model.predict(
                X_test.reshape(X_test.shape[0], X_test.shape[1], 1), verbose=0
            )
            cnn_pred = cnn_model.predict(X_test, verbose=0)
            transformer_pred = transformer_model.predict(X_test, verbose=0)
        elif self.framework == "pytorch":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
            X_test_lstm = X_test_torch.unsqueeze(-1)
            with torch.no_grad():
                ffnn_pred = ffnn_model(X_test_torch).cpu().numpy()
                lstm_pred = lstm_model(X_test_lstm).cpu().numpy()
                cnn_pred = cnn_model(X_test_torch).cpu().numpy()
                transformer_pred = transformer_model(X_test_torch).cpu().numpy()

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

        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(
            tk.END, f"Model Performance Metrics ({self.framework.capitalize()}):\n\n"
        )
        for model_name, (mse, mae) in metrics.items():
            self.metrics_text.insert(
                tk.END, f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}\n"
            )

    def plot_predictions(self):
        if not self.models:
            messagebox.showwarning("Warning", "Please train models first!")
            return

        try:
            sample_idx = int(self.sample_var.get())
            if sample_idx >= len(self.X_test):
                raise ValueError("Sample index out of range!")

            ffnn_model, lstm_model, cnn_model, transformer_model = self.models
            X_test, y_test = self.X_test, self.y_test
            sample = X_test[sample_idx].reshape(1, -1)

            if self.framework == "tensorflow":
                ffnn_pred = ffnn_model.predict(sample, verbose=0)[0]
                lstm_pred = lstm_model.predict(
                    sample.reshape(1, sample.shape[1], 1), verbose=0
                )[0]
                cnn_pred = cnn_model.predict(sample, verbose=0)[0]
                transformer_pred = transformer_model.predict(sample, verbose=0)[0]
            elif self.framework == "pytorch":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sample_torch = torch.tensor(sample, dtype=torch.float32).to(device)
                sample_lstm = sample_torch.unsqueeze(-1)
                with torch.no_grad():
                    ffnn_pred = ffnn_model(sample_torch).cpu().numpy()[0]
                    lstm_pred = lstm_model(sample_lstm).cpu().numpy()[0]
                    cnn_pred = cnn_model(sample_torch).cpu().numpy()[0]
                    transformer_pred = transformer_model(sample_torch).cpu().numpy()[0]

            self.ax.clear()
            self.ax.plot(y_test[sample_idx], label="Actual", linewidth=2)
            self.ax.plot(ffnn_pred, label="FFNN")
            self.ax.plot(lstm_pred, label="LSTM")
            self.ax.plot(cnn_pred, label="CNN")
            self.ax.plot(transformer_pred, label="Transformer")
            self.ax.set_title(
                f"Predictions vs Actual (Sample {sample_idx}, {self.framework.capitalize()})"
            )
            self.ax.set_xlabel("Hour")
            self.ax.set_ylabel("Scaled Load")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Plotting failed: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GridPredictionGUI(root)
    root.mainloop()
