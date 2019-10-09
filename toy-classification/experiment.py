import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bottleneck import Bottleneck
from utils import *
from pprint impot pprint

import argparse

parser = argparse.ArgumentParser(description='Stable Baseline RanDoom Experiment Specification')

parser.add_argument('--iterate_fpc', action='store_true',
                    help="Whether to vary over different numbers of functions per class")
parser.add_argument('--iterate_dps', action='store_true',
                    help="Whether to vary over different numbers of datapoint")

parser.add_argument('--dropout_rate', type=float,
                    help='Dropout_rate', default=0.5)
parser.add_argument('--vib_beta', type=float,
                    help='VIB beta', default=0.001)
parser.add_argument('--l2w_beta', type=float,
                    help='l2w beta', default=0.001)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
args = parser.parse_args()

assert args.iterate_fpc or args.iterate_dps
weight_decay = args.l2w_beta
beta = args.vib_beta
dropout_prob = args.dropout_rate

# number_datapoints = 100
latent_size = 1024
lr = 0.0001


cuda = True
datapoint_size = 100
num_classes = 5

# Iterate over:
regs = ['vib', 'weight', 'none', 'dropout']
seeds = list(range(10))
# num_epochs = 400
save_freq = 10
total_images_seen = 2048 * 400
prefix='RUN1'

if args.iterate_dps:
    # Vary number of datapoints for high number of funcs_per_class
    funcs_per_class = 64
    number_datapoints = None
    value_list = [64, 128, 256, 512, 1024, 2048]
    value_name = 'number_datapoints_train'
elif args.iterate_fpc:
    # Vary funcs_per_class for high number of datapoints
    funcs_per_class = None
    number_datapoints = 2048
    value_list = [2, 4, 8, 16, 32, 64]
    value_name = 'funcs_per_class'





kwargs = {
    'seed':None,
    'datapoint_size':datapoint_size,
    'number_datapoints_train':number_datapoints,
    'number_datapoints_test':200,
    'num_classes':num_classes,
    'funcs_per_class':funcs_per_class,
    'noise_strength':1.,
    'batch_size':25,
    'size_signal_area':20,
    'number_start_locations':3
}

print("Results dimensions: {}".format(results.shape))

for seed_idx, seed in enumerate(seeds):
    kwargs['seed'] = seed
    for value_idx, value in enumerate(value_list):
        kwargs[value_name] = value

        trainD, testD = create_dataset(
            **kwargs
        )

        num_epochs = total_images_seen // kwargs['number_datapoints_train']

        for reg in regs:
            results = np.zeros(shape=(2, num_epochs//save_freq))
            if reg == 'vib':
                wd = 0
                vib = True
                model_idx = 0
                dropout = 0.0
                reg_param = beta
            elif reg == 'weight':
                wd = weight_decay
                vib = False
                model_idx = 1
                dropout = 0.0
                reg_param = wd
            elif reg == 'none':
                wd = 0
                vib = False
                model_idx = 2
                dropout = 0.0
                reg_param = None
            elif reg == 'dropout':
                wd = 0
                vib = False
                model_idx = 3
                dropout = dropout_prob
                reg_param = dropout
            else:
                raise NotImplementedError()

            print("Running '{}'".format(reg))
            pprint(kwargs)

            c_train = []
            v_train = []
            c_test = []
            v_test = []
            dps_seen = []

            net = Net(vib=vib, data_size=datapoint_size, p_dropout=dropout, latent_size=latent_size, num_classes=num_classes)
            if cuda:
                net = net.cuda()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

            for epoch in range(num_epochs):
                running_c_loss = 0.0
                running_v_loss = 0.0
                nr_train_data = 0
                net = net.train()
                for i, data in enumerate(trainD, 0):
                    nr_train_data += len(data)

                    inputs, labels = data
                    if cuda:
                        inputs = inputs.cuda()
                        labels = labels.cuda()
                    optimizer.zero_grad()

                    outputs, kl = net(inputs)
                    classification_loss = criterion(outputs, labels)
                    loss = classification_loss + beta * kl
                    loss.backward()
                    optimizer.step()

                    running_c_loss += classification_loss.item()
                    running_v_loss += kl.item()

                if epoch % int(save_freq) == 0:
                    net = net.eval()
                    test_c_loss = 0.0
                    test_v_loss = 0.0
                    nr_test_data = 0
                    for i, data in enumerate(testD, 0):
                        nr_test_data += len(data)
                        inputs, labels = data
                        if cuda:
                            inputs = inputs.cuda()
                            labels = labels.cuda()
                        outputs, kl = net(inputs)
                        classification_loss = criterion(outputs, labels)
                        test_c_loss += classification_loss.item()
                        test_v_loss += kl.item()

                    dps_seen.append(epoch * kwargs['number_datapoints_train'])
                    c_train.append(running_c_loss/nr_train_data)
                    c_test.append(test_c_loss/nr_test_data)
                    v_train.append(running_v_loss/nr_train_data)
                    v_test.append(test_v_loss/nr_test_data)
                    print("\n{}\tTraining class loss: {}\t VIB: {}".format(
                        epoch,
                        running_c_loss/nr_train_data,
                        running_v_loss/nr_train_data))
                    print("{}\tTest class loss:     {}\t VIB: {}".format(
                        epoch,
                        test_c_loss/nr_test_data,
                        test_v_loss/nr_test_data))
                    results[0, epoch//save_freq] = running_c_loss/nr_train_data
                    results[1, epoch//save_freq] = test_c_loss/nr_test_data

            # 3: vib/weight/none
            # 3: train/test/samples_seen
            np.save("results/{}_reg{}_{}_DPs{}_fpc{}_seed{}_saveFreq{}".format(prefix,
                                                                               reg,
                                                                               reg_param,
                                                                               kwargs['number_datapoints_train'],
                                                                               kwargs['funcs_per_class'],
                                                                               kwargs['seed'],
                                                                               save_freq), results)

