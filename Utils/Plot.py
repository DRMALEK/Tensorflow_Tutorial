import matplotlib.pyplot as plt
import numpy as np


class Plot():
    @staticmethod
    def plot_history_line(x_label, y_label, ypoints, ypoints_label, ypoints2, ypoints2_label, figure_name):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot(ypoints, label=ypoints_label)
        plt.plot(ypoints2, label=ypoints2_label)
        plt.legend()

        plt.savefig(figure_name)
        plt.close()

    @staticmethod
    def plot_series(time, series, figure_name, format="-", start=0, end=None, on_same_figure=False):
        plt.plot(time[start:end], series[start:end], format)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)

        plt.savefig(figure_name)
        plt.close()