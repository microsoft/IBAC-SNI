import bottleneck as bn
bn.__version__ = '1.2.1'
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import re
from collections import defaultdict
import matplotlib
sns.set()
sns.set_style("ticks")
palette = sns.color_palette()

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

reg_names = {
    'vib': 'VIB',
    'none': 'NoReg',
    'dropout': 'Dropout',
    'weight': 'Weight Decay'
}

ma_length = 2

fixed_saveFreq = 10
train_test = 1

save_filename = "Vary_FPC.pdf"
fixed_nr_datapoints = 2048
fixed_fpcs = None

# save_filename = "Vary_nr_datapoints.pdf"
# fixed_nr_datapoints = None
# fixed_fpcs = 64

fixed_vib = 0.001
fixed_l2w = 0.001
fixed_dropout = 0.2
# fixed_vib = None
# fixed_l2w = None
# fixed_dropout = None

show_reg_value = False

mypath = "./results/"
from os import listdir
from os.path import isfile, join
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

dict = {
    'none': defaultdict(lambda: defaultdict(list)),
    'vib': defaultdict(lambda: defaultdict(list)),
    'weight': defaultdict(lambda: defaultdict(list)),
    'dropout': defaultdict(lambda: defaultdict(list))
}

x_label = r"Number of patterns $\omega^f$" if fixed_fpcs is None else "Number Datapoints"
y_label = "Loss"

for file in files:
    print(file)

    pattern = r'RUN[1-9]*_reg([a-z]+|none)_((?:[0-9]*[.])?[0-9]+|None|1e-[0-9]+)_DPs([0-9]+)_fpc([0-9]+)_seed([0-9]+)_saveFreq([0-9]+).npy'
    groups = re.match(pattern, file).groups()

    reg = groups[0]
    reg_value = float(groups[1]) if reg != 'none' else "None"
    nr_datapoints = int(groups[2])
    fpcs = int(groups[3])
    seed = int(groups[4])
    saveFreq = int(groups[5])

    if fixed_nr_datapoints is not None and fixed_nr_datapoints != nr_datapoints:
        continue

    if fixed_fpcs is not None and fixed_fpcs != fpcs:
        continue

    if reg == 'vib' and fixed_vib is not None and fixed_vib != reg_value:
        continue
    if reg == 'weight' and fixed_l2w is not None and fixed_l2w != reg_value:
        continue
    if reg == 'dropout' and fixed_dropout is not None and fixed_dropout != reg_value:
        continue

    data = np.load(join(mypath, file))[train_test]
    data = movingaverage(data, ma_length)

    if fixed_nr_datapoints is not None:
        # Iterate through fpcs
        dict[reg][reg_value][fpcs].append(data[-1])
    elif fixed_fpcs is not None:
        dict[reg][reg_value][nr_datapoints].append(data[-1])

fig, ax = plt.subplots(1, 1)
print("Number of colors: {}".format(len(palette)))
color_idx = -1
for reg_idx, reg in enumerate(dict):

    for reg_value in dict[reg]:
        color_idx += 1
        print("{}: Currently plotting: {}-{}".format(color_idx, reg, reg_value))
        x_values = []
        y_values = []
        y_stds = []
        for x_value in dict[reg][reg_value]:
            x_values.append(x_value)
            y_values.append(np.mean(dict[reg][reg_value][x_value]))
            N = len(dict[reg][reg_value][x_value])
            y_stds.append(np.std(dict[reg][reg_value][x_value]) / np.sqrt(N))

        x_values, y_values, y_stds = (list(t) for t in zip(*sorted(zip(x_values, y_values, y_stds))))
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        y_stds = np.array(y_stds)

        if show_reg_value:
            ax.plot(x_values, y_values, label=reg_names[reg] + " {}".format(reg_value), color=palette[color_idx])
        else:
            ax.plot(x_values, y_values, label=reg_names[reg], color=palette[color_idx])
        # ax.fill_between(x_values, y_values+y_stds, y_values-y_stds, alpha=0.5, edgecolor=None)
        ax.fill_between(x_values, y_values+y_stds, y_values-y_stds, alpha=0.5, color=palette[color_idx])

ax.legend()
ax.set_xscale('log', subsx=list(x_values)[::2])
ax.set_xticks(list(x_values)[::2])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.tick_params(
    which='major'
)
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)

fig.savefig(save_filename)



######### Multiple values

# for value_idx in range(nr_values):
#     for model_idx in range(nr_models):
#         last_index = np.argmax(arr[value_idx, model_idx, 0] == 0)
#         if last_index == 0:
#             last_index = -1
#             number_epochs = np.array(list(range(arr.shape[-1] - ma_length)))
#         else:
#             number_epochs = np.array(list(range(last_index - ma_length + 1)))
#         dps_seen = number_epochs * 10 * values[value_idx]
#         ax[value_idx,0].plot(dps_seen, movingaverage(arr[value_idx, model_idx, 0, :last_index], ma_length),
#                              label="Train "+regs[model_idx])
#         ax[value_idx,1].plot(dps_seen, movingaverage(arr[value_idx, model_idx, 1, :last_index], ma_length),
#                              label="Test "+regs[model_idx])
#     ax[value_idx, 0].set_ylabel("{}: {}".format(value_name, values[value_idx]))
# ax[0,0].legend()
# # ax[0,1].legend()
# fig.show()

