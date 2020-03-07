#!/usr/bin/env python3
"""
This script evaluates pre-trained models on 1, 2 and 3 Multiroom environments and plots the barplot from the paper.
"""

import argparse
import gym
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import os,sys,inspect
print(os.getcwd())
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = currentdir
print(parentdir)
sys.path.insert(0,parentdir+"/gym-minigrid/")
sys.path.insert(0,parentdir+"/torch_rl/")
import gym_minigrid

# Workaround: This is not my bottleneck, but some other library that throws an exception when loading pandas,
# complaining about bottleneck not having a version
import bottleneck
bottleneck.__version__ = "0.14"

import utils
import pandas as pd  # Import last. Not sure why but necessary
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()
sns.set_style("ticks")
fontsize = 18
params = {'legend.fontsize': fontsize, #'legend.handlelength': 2,
          'font.size': fontsize,
          'axes.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'xtick.labelsize': fontsize,
          'figure.figsize':(8.,6.)}
plt.rcParams.update(params)
sns.set_context("paper", rc=params)

# If you're evaluating a lot of experiments, it can take some time, so the evaluation results are also written into a
# CSV in the end. If 'read_csv==True', instead of re-evaluating everything it just reads the CSV
# Helpful for iterating over details in the plotting
read_csv = False

palette = sns.color_palette("colorblind")

envs = ['MiniGrid-MultiRoom-N1-v0', 'MiniGrid-MultiRoom-N2-v0', 'MiniGrid-MultiRoom-N3-v0', ]

# This dict contains all the different models, trained with different random seeds for each algorithm:
# {AlgorithmName: [List of different models trained with different random seeds]}

experiments = {
    "Test": ["N3r-vib1e6"]   # Here I trained something with --model N3r-vib1e6
}

# experiments = {
#     "IBAC":
#         ["{}-N3r-vib1e6".format(i) for i in range(25)] + \
#         ["0510-N3r-vib1e6", "0511-N3r-vib1e6", "0511-1-N3r-vib1e6", "0512-0-N3r-vib1e6"],
#     r"IBAC-SNI ($\lambda=0.5$)":
#         ["{}-N3r-vibS1e6".format(i) for i in range(25)] + \
#         ["0510-N3r-vibS1e6", "0511-N3r-vibS1e6", "0511-1-N3r-vibS1e6", "0512-0-N3r-vibS1e6", "0512-1-N3r-vibS1e6"],
#     "Dropout":
#         ["{}-N3r-dout0.2".format(i) for i in range(25)] + \
#         ["0510-N3r-dout0.2", "0511-N3r-dout0.2", "0511-1-N3r-dout0.2", "0512-0-N3r-dout0.2", "0512-1-N3r-dout0.2"],
#     r"Dropout-SNI ($\lambda=0.5$)":
#         ["{}-N3r-doutS0.2".format(i) for i in range(25)] + \
#         ["0510-N3r-doutS0.2", "0511-N3r-doutS0.2", "0511-1-N3r-doutS0.2", "0512-0-N3r-doutS0.2", "0512-1-N3r-doutS0.2"],
#     "NoReg":
#         ["{}-N3r-Plain".format(i) for i in range(25)] + \
#         ["0510-N3r-Plain", "0511-N3r-Plain", "0511-1-N3r-Plain", "0512-0-N3r-Plain", "0512-1-N3r-Plain"],
#     "Weight Decay":
#         ["{}-N3r-l2w1e4".format(i) for i in range(25)] + \
#         ["0510-N3r-l2w1e4", "0511-N3r-l2w1e4", "0511-1-N3r-l2w1e4", "0512-0-N3r-l2w1e4", "0512-1-N3r-l2w1e4"],
# }

# The keys here should match _exactly_ the keys in the 'experiments' dictionary
colors = {
    "Test": palette[0]
    # "IBAC": palette[9],
    # r"IBAC-SNI ($\lambda=0.5$)": palette[0],
    # r"IBAC-SNI ($\lambda=1$)": palette[4],
    # "NoReg": palette[2],
    # "Dropout": palette[3],
    # r"Dropout-SNI ($\lambda=0.5$)": palette[1],
    # r"Dropout-SNI ($\lambda=1$)": palette[8], # Was 6
    # "BatchNorm": palette[5],
    # "Weight Decay": palette[6],
}

nr_levels = 100
seed = 0
list_of_dicts = []

if not read_csv:
    for key_idx, key in enumerate(experiments):
        model_names = experiments[key]

        print(key)
        with tqdm(total=nr_levels * 3 * len(model_names)) as pbar:
            for model_idx, model_name in enumerate(model_names):

                for env_idx, env_name in enumerate(envs):
                    results = np.zeros((nr_levels,))
                    env = gym.make(env_name)
                    env.seed(0)
                    env = gym_minigrid.wrappers.FullyObsWrapper(env)

                    # Define agent

                    model_dir = utils.get_model_dir(model_name)
                    agent = utils.Agent(env_name, env.observation_space, model_dir, argmax=False)

                    lvl_cnt = 0

                    obs = env.reset()
                    while True:

                        action = agent.get_action(obs)
                        obs, reward, done, _ = env.step(action)
                        agent.analyze_feedback(reward, done)

                        if done:
                            results[lvl_cnt] = reward > 0
                            lvl_cnt += 1
                            pbar.update()
                            if lvl_cnt == nr_levels:
                                break
                            obs = env.reset()
                    list_of_dicts.append(
                            {
                                'key': key,
                                'model': model_name,
                                'rooms': "{} rooms".format(env_idx + 1),
                                'success_rate': np.mean(results)
                            }
                        )
    # #%%
    df = pd.DataFrame(list_of_dicts)
    print("Saving df")
    df.to_csv("barplot.csv")
else:
    df = pd.read_csv("barplot.csv")

# g = sns.catplot(x="rooms", y="success_rate", hue="key", data=df,
#                 height=6, kind="bar", palette="muted", ci=68)
g = sns.catplot(x="rooms", y="success_rate", hue="key", data=df,
                height=6., kind="bar", palette=colors, ci=68, legend_out=False,
                aspect=1.33)
for rooms in df['rooms'].unique():
    for key in df['key'].unique():
        print(rooms, key, df[ (df['key'] == key) & (df['rooms'] == rooms) ]['success_rate'].mean())
# g.despine(left=True)
g.set_ylabels("success rate")
g.axes[0,0].xaxis.set_label_text("")
# g.fig.legend(loc='upper right')
g.axes[0,0].spines['top'].set_visible(True)
g.axes[0,0].spines['right'].set_visible(True)
g.axes[0,0].legend().set_title('')

g.savefig("barplot.pdf")
g.savefig("barplot.png")

fig, ax = plt.subplots(figsize=(40,40))
g2 = sns.catplot(x='model', y="success_rate", data=df, kind='bar', palette='muted', ax=ax)
g2.despine(left=True)
g2.set_ylabels("success rate")
fig.savefig("performance.pdf")





