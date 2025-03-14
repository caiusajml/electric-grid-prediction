# Showcase Project: Electric Grid Anomaly Detection Model

This repository showcases a use of python, numpy, pandas, pytorch, tensorflow(keras).
The project is designed to train 4 models using Pytorch and tensorflow and to compare them against their capabilities to predict.
Without the acces to realdata this project resorted to generating synthetic healthy data using libraries in order to generate healthy electric grid data and train four different models.

# Electric Grid Prediction

**A machine learning project to predict electric grid load patterns using TensorFlow and PyTorch, featuring a user-friendly GUI for training, evaluation, and visualization.**

---

## Overview

`electric-grid-prediction` is a tool designed to forecast electric grid load over a 24-hour period using synthetic data. It leverages four distinct neural network architectures—Feedforward Neural Network (FFNN), Long Short-Term Memory (LSTM), Convolutional Neural Network (CNN), and Transformer—implemented in both TensorFlow and PyTorch frameworks. The project includes a Tkinter-based graphical user interface (GUI) that allows users to train models, evaluate their performance, and visualize predictions with ease.

This project demonstrates proficiency in:
- Libraries: numpy, pandas, pytorch, tensorflow(keras)
- Machine learning model design and implementation.
- Cross-framework development (TensorFlow and PyTorch).
- Data preprocessing and synthetic data generation.
- GUI development for interactive data science applications.

---

## Features

- **Dual Frameworks**: Train and evaluate models in TensorFlow or PyTorch, showcasing flexibility across popular ML ecosystems.
- **Model Variety**: Compare FFNN, LSTM, CNN, and Transformer architectures for time-series prediction.
- **Interactive GUI**: Train models, view performance metrics (MSE, MAE), and plot predictions with a few clicks.
- **Synthetic Data**: Generates realistic grid load data for testing (500 samples, 24 time steps).
- **Scalable Design**: Easily extensible for real-world data or additional models.

---

## Model Descriptions

This project utilizes several neural network models for prediction tasks. Here's a brief overview of each:

* **Recurrent Neural Networks (RNNs) & Long Short-Term Memory (LSTMs):**
    * RNNs are designed to process sequential data by maintaining an internal memory. LSTMs, a specialized type of RNN, address the vanishing gradient problem, enabling them to learn long-range dependencies.
    * **Use Case:** Best suited for sequential data, like time series. LSTMs, a type of RNN, excel at capturing long-term patterns in the data.
* **Convolutional Neural Networks (CNNs):**
    * CNNs use convolutional layers to automatically learn spatial or temporal hierarchies of features from data. In 1D applications, they slide filters across the sequence to detect local patterns.
    * **Use Case:** While commonly used for images, 1D CNNs can analyze time series data by identifying local patterns and features within the sequence.
* **Feedforward Neural Networks (FFNNs):**
    * FFNNs are the most basic type of neural network, where information flows in one direction, from input to output, through interconnected layers.
    * **Use Case:** A simpler model ideal for prediction tasks without strong time dependencies. Suitable when the data lacks significant sequential structure.
* **Transformers:**
    * Transformers rely on attention mechanisms to weigh the importance of different parts of the input sequence, allowing them to capture long-range dependencies effectively. They do not rely on reccurence.
    * **Use Case:** Powerful models that can effectively capture long-range dependencies in time series data. Often outperform LSTMs in complex applications.

---

## Installation

### Prerequisites
- Python 3.8
- Virtual environment (recommended)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/caiusajml/electric-grid-prediction.git
   cd electric-grid-prediction
   
2. **Create and Activate a Virtual Environment**:
   ```bash
   virtualenv  --python 3.8 venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   
3. **Install Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   

4. **Run the GUI**:
   ```bash
   python3 -m src.gui
   ```

---

## Usage
Running the GUI

Launch the application with:
   ```bash
   python3 -m src.gui
   ```
### GUI Walkthrough

1. **Select Framework: Choose "TensorFlow" or "PyTorch" from the dropdown.**
2. **Train Models: Click "Train Models" to train FFNN, LSTM, CNN, and Transformer on synthetic grid data (500 samples, 23 input steps → 23 output steps).**
3. **Evaluate Performance: Click "Evaluate & Show Metrics" to display MSE and MAE for each model.**
4. **Visualize Predictions: Enter a sample index (e.g., 0) and click "Plot Predictions" to see actual vs. predicted load curves for all models.**
5. **Exit: Close the GUI to end the session.**

## Project Structure

```
electric-grid-prediction/
├── README.md             # Project documentation
├── data/                 # Placeholder for future data files
├── models/               # Placeholder for saved models
├── requirements.txt      # Python dependencies
├── src/                  # Core source code
│   ├── __init__.py       # Package initialization
│   ├── data_generator.py # Synthetic grid data generation
│   ├── evaluate.py       # Model evaluation logic
│   ├── gui.py            # Tkinter GUI implementation
│   ├── models.py         # Model definitions (TF and PT)
│   └── train.py          # Training and preprocessing logic
└── tests/                # Unit tests
    └── test_models.py    # Model testing scripts
```

- **src/**: Contains the main source code.
  - **data_generator.py**: Creates synthetic load data with seasonal and random patterns.
  - **models.py**: Model (FFNN, LSTM, CNN, and Transformer) definitions for TensorFlow and PyTorch.
  - **train.py**: Handles data preprocessing and model training
  - **evaluate.py**: Computes and returns performance metrics.
  - **gui.py**: Tkinter-based GUI for user interaction, training, and plotting.
- **tests/**: Unit tests for model creation and data generation.

## Technical Highlights
- **Data Preprocessing:** Employs StandardScaler for normalization, ensuring consistent model performance.
- **Model Implementation:**
   - **TensorFlow:** Uses Sequential API for FFNN, LSTM, CNN, and Functional API for Transformer (adapted to avoid MultiHeadAttention issues).
   - **PyTorch:** Custom nn.Module classes with CPU/GPU support.
- **Key Challenges Solved:**
  - Overcame TensorFlow MultiHeadAttention softmax errors by redesigning the Transformer as a feedforward network.
  - Managed PyTorch dependency conflicts (e.g., typing-extensions) for seamless operation.
- **Performance:** TensorFlow CNN achieved an MSE of 0.0065, highlighting exceptional predictive accuracy.

## Future Enhancements
- **Real Data Support:** Integrate CSV or API-based grid data in data/.
- **Model Optimization:** Enhance CNN with additional layers or tune Transformer with Attention.
- **GUI Improvements:** Add progress bars, metric tables, and model export options.
- **Spinoff Project:** "Threat Anomaly Detection" using TensorFlow CNN to identify grid anomalies.

## Troubleshooting
** Tkinter Not Found (Especially on Ubuntu):

### If you encounter an error indicating that the tkinter module is not found, especially on Ubuntu or other Debian-based Linux distributions, you will likely need to install it manually. Tkinter is often packaged separately from the main Python installation.

### To install Tkinter on Ubuntu, use the following command:

```bash
sudo apt-get install python3-tk
```

### After installing Tkinter, try running the GUI again:

```bash
python3 -m src.gui.py
```

### This should resolve the issue.


## Author
## Kay Lancucki

## Email: kay.lancucki@gmail.com
### GitHub: github.com/caiusajml
## Crafted with precision to demonstrate machine learning and software engineering skills for real-world applications.
