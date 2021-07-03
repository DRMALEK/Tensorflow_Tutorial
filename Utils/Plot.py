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