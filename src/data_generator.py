import numpy as np
import pandas as pd


def generate_grid_data(n_samples=1000, time_steps=24):
    """
    Generates synthetic electric grid data with daily patterns, peaks, noise, and anomalies.

    Args:
        n_samples (int): Number of daily samples.
        time_steps (int): Number of time steps per sample (e.g., 24 for hourly data).

    Returns:
        reshaped_load (np.ndarray): Array of shape (n_samples, time_steps).
        data (pd.DataFrame): DataFrame with time, load, and day columns.
    """
    time = np.arange(n_samples * time_steps) / time_steps
    base_load = 100 + 50 * np.sin(2 * np.pi * time / 24)  # Daily sinusoidal pattern
    random_noise = np.random.normal(0, 10, n_samples * time_steps)
    peak_load = 30 * np.exp(-((time % 24) - 12) ** 2 / 10)  # Midday peak
    anomalies = np.zeros_like(time)
    anomalies[np.random.choice(len(time), size=int(0.01 * len(time)), replace=False)] = 50  # Anomalies
    load = np.maximum(0, base_load + peak_load + random_noise + anomalies)

    data = pd.DataFrame({'time': time, 'load': load})
    data['time'] = data['time'] % 24  # Hour of day
    data['day'] = np.floor(time / 24)  # Day number

    reshaped_load = data['load'].values.reshape(n_samples, time_steps)
    return reshaped_load, data