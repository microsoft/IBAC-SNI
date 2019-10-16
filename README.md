# Introduction 

This is the codebase for the NeurIPS 2019 paper "Generalization in Reinforcement Learning with Selective Noise
Injection and Information Bottleneck" which was work done by Maximilian Igl, Kamil Ciosek,
Yingzhen Li, Sebastian Tschiatschek, Cheng Zhang, Sam Devlin and Katja Hofmann during Maximilian's
internship at Microsoft Research Cambridge (UK).

It comprises several sub-projects:
1. `toy-classification` contains code for the classification experiment in the paper
2. `gym-minigrid` contains the grid-world environment and is adapted from
   https://github.com/maximecb/gym-minigrid (BSD 3-Clause license)
3. `torch_rl` contains the agent and training code to run on the `gym-minigrid` environment and is
   adapted from https://github.com/lcswillems/rl-starter-files (MIT license)
4. `coinrun` contains the code for the main results on the coinrun domain. It is code adapted from
   https://github.com/openai/coinrun (MIT license)

# Toy Classification Taks
To run the experiment, use
```
python experiment.py --iterate_fpc
```
or

```
python experiment.py --iterate_dps
```
. Using the addition `--cuda` uses GPUs. Hyperparameters for dropout, vib and l2w (=weight decay)
can be set using `--dropout_rate`, `--vib_beta` and `--l2w_beta`. Other hyperparameters like
learning rate, model size and number of epochs are hardcoded as variables in the python file.

For `--iterate_fpc`, the script iterates over the number of different patterns the encoding function
`f` uses, with values `[2, 4, 8, 16, 32, 64]`. For `--iterate_dps` it iterates over the number of
training data-points in `[64, 128, 256, 512, 1024, 2048]`. Both iterate over all 4 regularization
techniques `[VIB, L2W, Dropout, None]`. Results will be saved in `results/`, with one *.npy file for
each combination, with the used hyperparameters indicated in the filename.

## Plotting

To plot the results use 
```
python plots.py --plot_fc
```
or 
```
python plots.py --plot_dbs
```
and it will look through the `results/` folder for previously generated `*.npy` files. 
You can use `--fixed_vib_param`, `--fixed_dropout_param`, `--fixed_l2w_param`, to only plot the line
for one value instead of all available values of the respective regularization technique.

# Grid-world environment

First, install `gym-minigrid` with 
```
cd gym-minigrid
pip install -e .
```
The original gym-minigrid is modified by adding Multiroom environments `MiniGrid-MultiRoom-N2r-v0`
and `MiniGrid-MultiRoom-N3r-v0`, as well as `MiniGrid-Choice-9x9-v0` (which was not used, though).

Then also install the torch_rl:
```
cd ../torch_rl/torch_rl
pip install -e .
```

Furthermore, you'll need to set the "TORCH_RL_STORAGE" environmental variable, which determines
where the results will be stored, e.g. by including `export TORCH_RL_STORAGE=~/results` in your `~/.bashrc`.

The results from the paper can then be reproduced by running from the (outer!) `torch_rl` directory:
```
python -m scripts.train --frames 100000000 --algo ppo --env MiniGrid-MultiRoom-N3r-v0 --model N3r-vib1e6 --save-interval 100 --tb --fullObs --model_type default2 --use_bottleneck --beta 0.000001
python -m scripts.train --frames 100000000 --algo ppo --env MiniGrid-MultiRoom-N3r-v0 --model N3r-vibS1e6 --save-interval 100 --tb --fullObs --model_type default2 --use_bottleneck --beta 0.000001 --sni_type vib
python -m scripts.train --frames 100000000 --algo ppo --env MiniGrid-MultiRoom-N3r-v0 --model N3r-Plain --save-interval 100 --tb --fullObs --model_type default2
python -m scripts.train --frames 100000000 --algo ppo --env MiniGrid-MultiRoom-N3r-v0 --model N3r-l2w1e4 --save-interval 100 --tb --fullObs --model_type default2 --use_l2w --beta 0.0001
python -m scripts.train --frames 100000000 --algo ppo --env MiniGrid-MultiRoom-N3r-v0 --model N3r-dout0.2 --save-interval 100 --tb --fullObs --model_type default2 --use_dropout 0.2
python -m scripts.train --frames 100000000 --algo ppo --env MiniGrid-MultiRoom-N3r-v0 --model N3r-doutS0.2 --save-interval 100 --tb --fullObs --model_type default2 --use_dropout 0.2 --sni_type dropout
```
When using multiple runs, make sure to change the `--model` name, which determines the folder name
of the results and also make sure to specify different random seeds using `--seed <nr>`.

## Plotting

To plot the results, modify the `plots.py` file by changing the `path`, as well as the `experiments`
dictionary to specify which subfolders in `path` you would like to plot.

# Coinrun

Please follow the installation instructions taken from the original repo to install the requirements: 
```
# Linux
apt-get install mpich build-essential qt5-default pkg-config
# Mac
brew install qt open-mpi pkg-config

cd coinrun
pip install tensorflow==1.12.0  # or tensorflow-gpu
pip install -r requirements.txt
pip install -e .
```

Also, in `coinrun/coinrun/config.py` set the `self.WORKDIR` and `self.TB_DIR` variables.

## Reproducing Results

To reproduce the results, run on a NC24 with 4 GPUs (3 will be used for training, one for testing):
```
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --run-id baseline --num-levels 500 --test --long --l2 0.0001 -uda 1
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --run-id ibac-sni-lambda0.5 --num-levels 500 --test --long --l2 0.0001 -uda 1 --beta 0.0001 --nr-samples 12 --sni
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --run-id ibac-sni-lambda1.0 --num-levels 500 --test --l2 0.0001 -uda 1 --beta-l2a 0.0001 --long
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --run-id ibac --num-levels 500 --test --long --l2 0.0001 -uda 1 --beta 0.0001 --nr-samples 12
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --run-id dropout0.2-sni-lambda0.5 --num-levels 500 --test --long --l2 0.0001 -uda 1 --dropout 0.2 --sni2
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --run-id dropout0.2-sni-lambda1.0 --num-levels 500 --test --long --l2 0.0001 -uda 1 --dropout 0.2 --openai
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --run-id dropout0.2 --num-levels 500 --test --long --l2 0.0001 -uda 1 --dropout 0.2
RCALL_NUM_GPU=4 mpiexec -n 4 python3 -m coinrun.train_agent --run-id batchnorm --num-levels 500 --test --long --l2 0.0001 -uda 1 -norm 1
```

where all the results are including weight decay (`--l2 0.0001`) and data augmentation (`-uda 1`). 
Batchnorm is `-norm 1`, Dropout is `--dropout 0.2`, VIB is `--beta 0.0001`, L2 on Activations is
`--beta-l2a 0.0001` which corresponds to VIB-SNI with `lambda=1`. For dropout, we can either use SNI
with `lambda=0.5` by using `--sni2` or with `lamda=1.0` by using `--openai`.

The experiments, especially with the `--long` flag, take a while. If it's run on the VMs, it will
likely crash at some point (around 6pm is particularly likely), probably because the servers are
preemtible.
If they do, you can restart with the additional arguments `--restore-id <run-id>` and
`--restore-step <step>` where you can read out the step from the tensor-board plot.

## Plotting

A note on the tensorboard plots: For each run, you will see 4 different folders 'name_0', 'name_1',
etc..
The 'name_0' version is the performance on the training set. The 'name_1' version is the performance
on the test set.
Furthermore, to compare to the paper you'll need to multiply the number of frames by 3, as tensorboard reports
the frames _per worker_, whereas the paper reports the total number of frames used for training.


Using `plots.py`, fill in the `path` variable, as well as `plotname`, `plotname_kl` and the
`experiments` dictionary where each entry corresponds to one line which will be the average over all
run-ids listed in the corresponding list (see the script for examples.)
