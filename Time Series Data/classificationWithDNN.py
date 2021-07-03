import sys
sys.path.append('../')
from matplotlib import pyplot as plt
from Utils.Plot import Plot
from Utils.Series import Series
import numpy as np
import tensorflow as tf

# Craete a simple series
time_seq = np.arange(4 * 365 + 1, dtype="float32")
series = Series.trend(time_seq, 0.1)
baseline = 10
amplitude = 20
slope = 0.09
noise_level = 5

series = baseline + Series.trend(time_seq, slope)
series += Series.seasonality(time_seq, period=365, amplitude=amplitude)
series += Series.noise(time_seq, noise_level, seed=42)

# Create a windowed dataset from the series
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000
dataset = Series.windowed_dataset(
    series, window_size, batch_size, shuffle_buffer_size)

# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

# MSE: mean squared error and SGD (stochastic gradient descent)
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(
    lr=1e-6, momentum=0.9))

# Train the model
model.fit(dataset, epochs=100, verbose=1)

# Do evaluation for a single value
start_point = 1000
print(series[start_point:start_point+window_size])
print(series[start_point+window_size])
print(model.predict(series[start_point:start_point+window_size][np.newaxis]))

# Do evaluation over a set of values
forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[start_point-window_size:]
results = np.array(forecast)[:, 0, 0]

