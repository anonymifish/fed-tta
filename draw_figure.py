import numpy as np
import matplotlib.pyplot as plt

x1 = [20, 33, 51, 79, 101, 121, 132, 145, 162, 182, 203, 219, 232, 243, 256, 270, 287, 310, 325]
y1 = [49, 48, 48, 48, 48, 87, 106, 123, 155, 191, 233, 261, 278, 284, 297, 307, 341, 319, 341]
x2 = [31, 52, 73, 92, 101, 112, 126, 140, 153, 175, 186, 196, 215, 230, 240, 270, 288, 300]
y2 = [48, 48, 48, 48, 49, 89, 162, 237, 302, 378, 443, 472, 522, 597, 628, 661, 690, 702]
x3 = [30, 50, 70, 90, 105, 114, 128, 137, 147, 159, 170, 180, 190, 200, 210, 230, 243, 259, 284, 297, 311]
y3 = [48, 48, 48, 48, 66, 173, 351, 472, 586, 712, 804, 899, 994, 1094, 1198, 1360, 1458, 1578, 1734, 1797, 1892]
x = np.arange(20, 350)

aux_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
aux_ratio_no_shift = [68.23, 66.63, 68.05, 67.87, 68.62, 70.04, 69.61, 69.34, 69.35, 70.77]
aux_ratio_contrast = [27.15, 27.50, 28.29, 28.48, 28.50, 28.81, 28.75, 28.67, 28.25, 28.48]
aux_ratio_label_shift = [43.67, 46.81, 47.44, 47.65, 47.35, 45.66, 45.93, 45.52, 45.59, 44.89]
aux_ratio_avg1 = [60.56, 58.92, 60.33, 59.74, 61.21, 62.56, 62.36, 61.79, 62.10, 62.87]
aux_ratio_avg2 = [38.73, 42.01, 42.16, 42.75, 42.34, 40.59, 41.04, 40.56, 40.66, 39.58]
aux_ratio_avg = [47.67, 48.37, 49.25, 49.30, 49.60, 49.53, 49.54, 49.18, 49.19, 49.32]

threshold = [1, 5, 10, 15, 20, 25, 30]
threshold_no_shift = [44.92, 46.54, 54.36, 64.12, 68.17, 68.75, 64.31]
threshold_contrast = [27.26, 27.38, 25.14, 19.77, 19.84, 27.34, 28.33]
threshold_label_shift = [38.34, 36.45, 40.32, 44.11, 46.91, 47.12, 48.05]
threshold_avg1 = [43.64, 45.21, 51.15, 60.07, 61.97, 59.32, 55.06]
threshold_avg2 = [35.71, 35.51, 37.46, 40.82, 41.29, 41.88, 43.27]
threshold_avg = [37.97, 38.22, 41.69, 45.78, 47.64, 48.88, 47.80]

filter = [1, 5, 10, 20, 50, 100]
filter_no_shift = [54.94, 67.52, 68.4, 68.75, 68.51, 68.55]
filter_contrast = [28.63, 27.34, 27.34, 27.34, 27.34, 27.34]
filter_label_shift = [48.66, 48.16, 47.36, 47.12, 47.2, 47.19]
filter_avg1 = [47.28, 59.09, 59.29, 59.32, 59.33, 59.32]
filter_avg2 = [43.51, 42.24, 41.95, 41.88, 41.86, 41.86]
filter_avg = [44.60, 48.87, 48.87, 48.88, 48.85, 48.85]

for name in ['aux_ratio', 'threshold', 'filter']:
    fig, ax = plt.subplots(2, 3, sharex=True, sharey='none', figsize=(30, 20))
    for one in ax.flat:
        one.tick_params(axis='both', labelsize=30)
    ax[0, 0].plot(eval(name), eval(f'{name}_no_shift'), linewidth=10.0, marker='s', markersize=20.0)
    ax[0, 0].set_title('no shift', fontsize=40)
    ax[0, 1].plot(eval(name), eval(f'{name}_contrast'), linewidth=10.0, marker='s', markersize=20.0)
    ax[0, 1].set_title('contrast corrupt', fontsize=40)
    ax[0, 2].plot(eval(name), eval(f'{name}_label_shift'), linewidth=10.0, marker='s', markersize=20.0)
    ax[0, 2].set_title('label shift', fontsize=40)
    ax[1, 0].plot(eval(name), eval(f'{name}_avg1'), linewidth=10.0, marker='s', markersize=20.0)
    ax[1, 0].set_title('covaraite shift', fontsize=40)
    ax[1, 1].plot(eval(name), eval(f'{name}_avg2'), linewidth=10.0, marker='s', markersize=20.0)
    ax[1, 1].set_title('hybrid shift', fontsize=40)
    ax[1, 2].plot(eval(name), eval(f'{name}_avg'), linewidth=10.0, marker='s', markersize=20.0)
    ax[1, 2].set_title('average', fontsize=40)

    plt.savefig(f'./test_{name}.png')
