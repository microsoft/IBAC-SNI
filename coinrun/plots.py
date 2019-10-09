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

# window_size is used for the moving average
window_size = 2000
xlims = None
max_step = 200000000
tag = "rew_mean"
ylims_test = (6, 8.1)
ylims_train = (8, 10)
path = "/mnt/c/Users/t-maigl/results/tb_log/{}/"


######## Compare Dropouts
# plotname = "Paper_Coinrun_Dropout.pdf"
# plotname_kl = "Paper_Coinrun_Dropout_KL.pdf"
# experiments = {
#     "Baseline": [
#         '0501_0_l2w_uda_{}', '0501_1_l2w_uda_{}'
#     ],
#     "VIB-SNI": [
#         '0501_0_vibnn12e4_l2w_uda_{}', '0501_1_vibnn12e4_l2w_uda_{}', '0424_vibnn12e4_l2w_uda_{}'
#     ],
#
#     "Dropout p=0.1": [
#         "0514_l2w_uda_dropout0.1_{}"
#     ],
#     "Dropout-Openai p=0.1": [
#         "0509_l2w_uda_dropoutOpenai0.1_{}"
#     ],
#     "Dropout-SNI p=0.1": [
#         "0514_l2w_uda_dropout0.1_sni2new_{}"
#     ],
#
#     "Dropout p=0.2": [
#         "0506_l2w_uda_dropout0.2_{}"
#     ],
#     "Dropout-Openai p=0.2": [
#         "0509_l2w_uda_dropoutOpenai0.2_{}"
#     ],
#     "Dropout-SNI p=0.2": [
#         "0514_l2w_uda_dropout0.2_sni2new_{}"
#     ],
#
#     "Dropout p=0.5": [
#         "0506_l2w_uda_dropout0.5_{}"
#     ],
#     "Dropout-Openai p=0.5": [
#         "0509_l2w_uda_dropoutOpenai0.5_{}"
#     ],
# }
# experiments = {
#
#     "L2A": [
#         '0416_l2a_l2w_uda_{}',
#     ],
#     "Dropout-Det": [
#         "0509_l2w_uda_dropoutOpenai0.2_{}"
#     ],
#
# }
#

# plotname = "Dropout_on_Plain.pdf"
# plotname_kl = "Dropout_on_Plain_KL.pdf"
# experiments = {
#     r"IBAC-SNI ($\lambda=0.5$)": [
#         '0424_vibnn12e4_{}'
#     ],
#     "IBAC": [
#         '0519_vib12e4_{}'
#     ],
#     r"Dropout-SNI ($\lambda=0.5$)": [
#         '0514_dropout0.2_sni2new_{}', '0519_dropout0.2_sni2new_{}'
#     ],
#     r"Dropout-SNI ($\lambda=1$)": [
#         '0519_dropoutOpenai0.2_{}', '0519_2_dropoutOpenai0.2_{}'
#     ],
#     "Dropout": [
#         '0514_dropout0.2_{}', '0519_dropout0.2_{}'
#     ],
# }

###### Dropout Comparison Appendix
plotname = "Paper_Dropout_App.pdf"
plotname_kl = "Paper_Dropout_App_KL.pdf"
experiments = {
    "Baseline": [
        '0501_0_l2w_uda_{}', '0501_1_l2w_uda_{}'
    ],
    r"Dropout-SNI ($\lambda=0.5$)": [
        "0514_l2w_uda_dropout0.2_sni2new_{}", "0519_l2w_uda_dropout0.2_sni2_{}"
    ],
    r"Dropout-SNI ($\lambda=1$)": [
        "0519_l2w_uda_dropoutOpenai0.2_{}"
    ],
    "Dropout": [
        "0506_l2w_uda_dropout0.2_{}", "0519_l2w_uda_dropout0.2_{}", "0519_2_l2w_uda_dropout0.2_{}"
    ],
}


##### Main Train/Test Plot
plotname = "Paper_Coinrun_Main.pdf"
plotname_kl = "Paper_Coinrun_KL.pdf"
experiments = {
    r"IBAC-SNI ($\lambda=0.5$)": [
        '0501_0_vibnn12e4_l2w_uda_{}', '0501_1_vibnn12e4_l2w_uda_{}', '0424_vibnn12e4_l2w_uda_{}', '0514_vibnn12e4_l2w_uda_{}'
    ],
    r"IBAC-SNI ($\lambda=1$)": [
        '0416_l2a_l2w_uda_{}'
    ],
    "IBAC": [
        "0506_l2w_uda_vib12e4_{}", "0519_l2w_uda_vib12e4_{}"
    ],
    "Baseline": [
        '0501_0_l2w_uda_{}', '0501_1_l2w_uda_{}'
    ],
    r"Dropout-SNI ($\lambda=0.5$)": [
        "0514_l2w_uda_dropout0.2_sni2new_{}", "0519_l2w_uda_dropout0.2_sni2_{}"
    ],
    r"Dropout-SNI ($\lambda=1$)": [
        "0519_l2w_uda_dropoutOpenai0.2_{}"
    ],
    "Dropout": [
        "0506_l2w_uda_dropout0.2_{}", "0519_l2w_uda_dropout0.2_{}", "0519_2_l2w_uda_dropout0.2_{}"
    ],
    "BatchNorm":[
        "0506_l2w_uda_bn_{}", "0519_l2w_uda_bn_{}"
    ],
}

############# For KL
# plotname = "Paper_Coinrun_Main.pdf"
# plotname_kl = "Paper_Coinrun_KL.pdf"
# experiments = {
#     # "PVIB": [
#     #     "0506_l2w_uda_vib12e4_{}"
#     # ],
#     r"IBAC-SNI ($\lambda=0.5$)": [
#         '0514_vibnn12e4_l2w_uda_{}'
#     ],
#     # "Dropout": [
#     #     "0506_l2w_uda_dropout0.2_{}"
#     # ],
#     # "Dropout-Det": [
#     #     "0509_l2w_uda_dropoutOpenai0.2_{}"
#     # ],
#     r"Dropout-SNI ($\lambda=0.5$)": [
#         "0514_l2w_uda_dropout0.2_sni2new_{}", "0519_l2w_uda_dropout0.2_sni2_{}"
#     ],
# }

fig_main, ax_main = plt.subplots(1, 1)
fig_main2, ax_main2 = plt.subplots(1, 1)
fig_approxkl, ax_approxkl = plt.subplots(1, 1)
palette = sns.color_palette()

for ending, marker in zip([0, 1], ['--', '-']):
    for key_idx, key in enumerate(experiments):
        all_steps = []
        all_values = []
        all_approxkl_run = []
        all_approxkl_train = []
        for idx in range(len(experiments[key])):
            dirname = experiments[key][idx].format(ending)
            print(dirname)
            steps = []
            values = []
            approxkl_run = []
            approxkl_train = []
            modified_path = path.format(dirname)
            for filename in os.listdir(modified_path):
                if not filename.startswith('events'):
                    continue
                try:
                    for e in tf.train.summary_iterator(modified_path + filename):
                        for v in e.summary.value:
                            if v.tag == 'rew_mean' and e.step <= max_step:
                                steps.append(e.step)
                                values.append(v.simple_value)
                            if ending == 0:
                                if v.tag == 'approxkl_run' and e.step <= max_step:
                                    approxkl_run.append(v.simple_value)
                                elif v.tag == 'approxkl_train' and e.step <= max_step:
                                    approxkl_train.append(v.simple_value)
                except:
                    pass
                # print(e)
            steps = np.array(steps)[window_size//2:-window_size//2] * 3
            values = movingaverage(np.array(values), window_size)
            min_len = min(steps.shape[0], values.shape[0])
            values, steps = values[:min_len], steps[:min_len]

            if len(approxkl_run) == 0:
                approxkl_run = np.zeros(min_len)
                approxkl_train = np.zeros(min_len)
            else:
                approxkl_run = movingaverage(np.array(approxkl_run), window_size)[:min_len]
                approxkl_train = movingaverage(np.array(approxkl_train), window_size)[:min_len]
            all_steps.append(steps)
            all_values.append(values)
            all_approxkl_run.append(approxkl_run)
            all_approxkl_train.append(approxkl_train)

        min_length = np.inf
        for steps, values in zip(all_steps, all_values):
            min_length = min(min_length, steps.shape[0])
            min_length = min(min_length, values.shape[0])
        new_all_steps = []
        new_all_values = []
        new_all_approxkl_run = []
        new_all_approxkl_train = []
        for steps, values, approxkl_run, approxkl_train in zip(all_steps, all_values, all_approxkl_run, all_approxkl_train):
            new_all_steps.append(steps[:min_length])
            new_all_values.append(values[:min_length])
            new_all_approxkl_run.append(approxkl_run[:min_length])
            new_all_approxkl_train.append(approxkl_train[:min_length])

        all_steps = np.stack(new_all_steps)
        all_values = np.stack(new_all_values)
        all_approxkl_run = np.stack(new_all_approxkl_run)
        all_approxkl_train = np.stack(new_all_approxkl_train)

        mean = np.mean(all_values, 0)[::10]
        std = np.std(all_values, 0)[::10]
        steps = all_steps[0][::10]

        # label = key if ending == 0 else None
        label = key
        if ending == 0:
            ax = ax_main
        else:
            ax = ax_main2

        # ax.plot(steps, mean, label=label, linestyle=marker, color=palette[key_idx])
        ax.plot(steps, mean, label=label, color=palette[key_idx])
        ax.fill_between(steps, mean+std, mean-std, alpha=0.5, color=palette[key_idx])

        if ending == 0 and all_approxkl_run[0,0] != 0:
            mean = np.mean(all_approxkl_run, 0)[::10]
            std = np.std(all_approxkl_run, 0)[::10]
            ax_approxkl.plot(steps, mean, label=key + " (det)", color=palette[key_idx])
            ax_approxkl.fill_between(steps, mean+std, mean-std, alpha=0.5, color=palette[key_idx])

            mean = np.mean(all_approxkl_train, 0)[::10]
            std = np.std(all_approxkl_train, 0)[::10]
            ax_approxkl.plot(steps, mean, label=key + " (stoch)", linestyle='--', color=palette[key_idx])
            ax_approxkl.fill_between(steps, mean+std, mean-std, alpha=0.5, color=palette[key_idx])
        elif ending == 0 and all_approxkl_train[0,0] != 0:
            mean = np.mean(all_approxkl_train, 0)[::10]
            std = np.std(all_approxkl_train, 0)[::10]
            ax_approxkl.plot(steps, mean, label=key, linestyle='-.', color=palette[key_idx])
            ax_approxkl.fill_between(steps, mean+std, mean-std, alpha=0.5, color=palette[key_idx])

ax_main.legend(loc='upper left')
ax_main2.legend(loc='upper left')
ax_main2.set_ylim(*ylims_test)

# if plotname == "Dropout_on_Plain.pdf":
#     ax_main2.set_ylim(5, 7.1)
ax_main.set_ylim(*ylims_train)

ax_main.set_xlabel("Frames")
ax_main2.set_xlabel("Frames")
ax_main.set_ylabel("Return")
ax_main2.set_ylabel("Return")
# if xlims is not None:
#     ax_main.set_xlim(*xlims)
fig_main.savefig("Train_"+plotname, bbox_inches='tight')
fig_main2.savefig("Test_"+plotname, bbox_inches='tight')

ax_approxkl.legend(loc='upper right')
ax_approxkl.set_xlabel("Frames")
ax_approxkl.set_ylabel("Approx KL")
fig_approxkl.savefig(plotname_kl, bbox_inches='tight')

