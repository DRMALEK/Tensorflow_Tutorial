import numpy as np
import tensorflow as tf
from statistics import median

class Series:
    @staticmethod
    def trend(time, slope=0):
        return slope * time

    @staticmethod
    def seasonal_pattern(season_time):
        return np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))

    @staticmethod
    def seasonality(time, period, amplitude=1, phase=0):
        season_time = ((time + phase) % period) / period
        return amplitude * Series.seasonal_pattern(season_time)

    @staticmethod
    def noise(time, noise_level=1, seed=None):
        rnd = np.random.RandomState(seed)
        return rnd.randn(len(time)) * noise_level

    @staticmethod
    def windowed_dataset(series, window_size, batch_size, shuffle_buffer, expand_dim=False):
        # In case of using convolutional neural network, convert the series to a 1D tensor
        if expand_dim:
            series = tf.expand_dims(series, axis=-1)
            print(series.shape)
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(window_size + 1, shift=1,
                                 drop_remainder=True)
        dataset = dataset.flat_map(
            lambda window: window.batch(window_size + 1))
        dataset = dataset.shuffle(shuffle_buffer).map(
            lambda window: (window[:-1], window[-1]))
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset
    
    @staticmethod
    def get_data(data_path):
        f = open(data_path)
        data = f.read()
        f.close()
        lines = data.split('\n')
        header = lines[0].split(',')
        lines = lines[1:]
        temperatures = []
        # Loop over the data, and make it as a series of points
        for line in lines:
            if line:
                linedata = line.split(',')
                linedata = linedata[1:13]
                linedata = [float(i) for i in linedata]

                # Skip the croupted rows (all items are 999.9 or 999.90)
                croupted_data = True
                for i in linedata:
                    if i == 999.90 or i == 999.9:
                        croupted_data = True
                    else:
                        croupted_data = False
                if croupted_data:
                    continue
                
                for item in linedata:
                    if item == 999.90 or item == 999.9:
                        item = median(linedata)   # if it is croupted, take the median of the row
                    temperatures.append(item)

        series = np.asarray(temperatures)
        time = np.arange(len(temperatures), dtype="float32")
        return time, series
