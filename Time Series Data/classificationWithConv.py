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
    series, window_size, batch_size, shuffle_buffer_size, expand_dim=True)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1,
                           padding="causal", activation="relu", input_shape=[None, 1]),
    tf.keras.layers.Dense(28, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
])
optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.5)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=100, verbose=1)

# Forecasting function for an entire series
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


forecast = model_forecast(model, series[..., np.newaxis], window_size)

# Get just the validaiton result
results = forecast[split_time - window_size:-1, -1, 0]

# Print the MAE score ...
tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()