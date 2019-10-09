import bottleneck as bn
bn.__version__ = '1.2.1'
import tensorflow as tf
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
sns.set_style("ticks")

params = {'legend.fontsize': 10, 'legend.handlelength': 2,
          'font.size': 10}
plt.rcParams.update(params)

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

plotname = "Paper_N3r_stdErr_median.pdf"
agg_fn = np.mean
# plotname = "Paper_N3r.pdf"
window_size = 100
plot_std_err = True
# ylims = (5.5, 10)
# ylims = (4, 10)
xlims = None
max_step = 200000000
path = "/home/t-maigl/results-cr/minigrid/{}/"


experiments = {
    "NoReg": [
        "0510-N3r-Plain", "0511-N3r-Plain", "0511-1-N3r-Plain", "0512-0-N3r-Plain", "0512-1-N3r-Plain",
    ],
    "Weight Decay": [
        "0510-N3r-l2w1e4", "0511-N3r-l2w1e4", "0511-1-N3r-l2w1e4", "0512-0-N3r-l2w1e4", "0512-1-N3r-l2w1e4",
    ],
    "Dropout": [
        "0510-N3r-dout0.2", "0511-N3r-dout0.2", "0511-1-N3r-dout0.2", "0512-0-N3r-dout0.2", "0512-1-N3r-dout0.2",
    ],
    r"Dropout-SNI ($\lambda=0.5$)": [
        "0510-N3r-doutS0.2", "0511-N3r-doutS0.2", "0511-1-N3r-doutS0.2", "0512-0-N3r-doutS0.2", "0512-1-N3r-doutS0.2",
    ],
    # "Dropout p=0.5": [
    #     "0510-N3r-dout0.5", "0511-N3r-dout0.5", "0511-1-N3r-dout0.5", "0512-0-N3r-dout0.5", "0512-1-N3r-dout0.5",
    # ],
    # "Dropout (SNI2) p=0.5": [
    #     "0510-N3r-doutS0.5", "0511-N3r-doutS0.5", "0511-1-N3r-doutS0.5", "0512-0-N3r-doutS0.5", "0512-1-N3r-doutS0.5",
    # ],
    "IBAC": [
        "0510-N3r-vib1e6", "0511-N3r-vib1e6", "0511-1-N3r-vib1e6", "0512-0-N3r-vib1e6", "0512-1-N3r-vib1e6",
    ],
    r"IBAC-SNI ($\lambda=0.5$)": [
        "0510-N3r-vibS1e6", "0511-N3r-vibS1e6", "0511-1-N3r-vibS1e6", "0512-0-N3r-vibS1e6", "0512-1-N3r-vibS1e6",
    ],
}

fig_main, ax_main = plt.subplots(1,1)
palette = sns.color_palette()

for key_idx, key in enumerate(experiments):
    print(key)
    all_steps = []
    all_values = []
    for idx in range(len(experiments[key])):
        dirname = experiments[key][idx]
        print(dirname)
        steps = []
        values = []
        modified_path = path.format(dirname)
        for filename in os.listdir(modified_path):
            if not filename.startswith('events'):
                continue
            try:
                for e in tf.train.summary_iterator(modified_path + filename):
                    for v in e.summary.value:
                        if v.tag == 'return_mean' and e.step <= max_step:
                            steps.append(e.step)
                            values.append(v.simple_value)
            except:
                pass
            # print(e)
        steps = np.array(steps)[window_size//2:-window_size//2]
        values = movingaverage(np.array(values), window_size)
        min_len = min(steps.shape[0], values.shape[0])
        values, steps = values[:min_len], steps[:min_len]

        all_steps.append(steps)
        all_values.append(values)

    min_length = np.inf
    for steps, values in zip(all_steps, all_values):
        min_length = min(min_length, steps.shape[0])
        min_length = min(min_length, values.shape[0])
    new_all_steps = []
    new_all_values = []
    for steps, values in zip(all_steps, all_values):
        new_all_steps.append(steps[:min_length])
        new_all_values.append(values[:min_length])

    all_steps = np.stack(new_all_steps)
    all_values = np.stack(new_all_values)

    num_trajectories = all_values.shape[0]
    print("Number trajectoriesf: {}".format(num_trajectories))
    mean = agg_fn(all_values, 0)[::100]
    std = np.std(all_values, 0)[::100] / np.sqrt(num_trajectories)
    steps = all_steps[0][::100]
    print(mean.shape)
    ax_main.plot(steps, mean, label=key, color=palette[key_idx])
    if plot_std_err:
        # ax_main.fill_between(steps, mean+std, mean-std, alpha=0.5, color=palette[key_idx])
        ax_main.fill_between(steps, mean+std, mean-std, alpha=0.5, color=palette[key_idx])

ax_main.legend(loc='upper left')
ax_main.set_xlabel("Frames")
ax_main.set_ylabel("Return")
# ax_main.set_ylim(*ylims)
# if xlims is not None:
#     ax_main.set_xlim(*xlims)
fig_main.savefig(plotname, bbox_inches='tight')


