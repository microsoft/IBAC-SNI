import numpy as np
import torch.utils.data as utils
import torch
import torch.nn as nn
from bottleneck import Bottleneck
import matplotlib.pyplot as plt

def create_function(order, max_coef):
    a = np.random.uniform(-max_coef, max_coef, size=order)
    b = np.random.uniform(-max_coef, max_coef, size=order)
    def function(x):
        function_value = 0
        for n in range(len(a)):
            function_value += a[n] * np.cos((n+1)*x) + b[n] * np.sin((n+1)*x)
        return function_value
    return function

#
def normalize(y):
    return (y - np.mean(y)) / np.std(y)

# fig, ax = plt.subplots(5, 5)
# for i in range(5):
#     for j in range(5):
#         func = create_function(3, 1)
#         x = np.linspace(0,2 * np.pi)
#         # ax[i,j].plot(x, normalize(func(x)))
#         ax[i,j].plot(x, func(x))
# plt.show()


def create_dataset(seed, datapoint_size, number_datapoints_train, number_datapoints_test, num_classes, funcs_per_class, noise_strength, batch_size, size_signal_area, number_start_locations):
    np.random.seed(seed)
    functions_train = [[create_function(3,1) for i in range(funcs_per_class)] for j in range(num_classes)]
    functions_test = [[create_function(3,1) for i in range(funcs_per_class)] for j in range(num_classes)]

    pattern_x = np.linspace(0, 2*np.pi, size_signal_area)
    signals = [create_function(3,1)(pattern_x) for i in range(num_classes)]

    classes_train = np.random.randint(0, num_classes, size=number_datapoints_train)
    classes_test = np.random.randint(0, num_classes, size=number_datapoints_test)

    possible_x = np.linspace(0, 2*np.pi, 2000)
    x = np.array(sorted(np.random.choice(possible_x, datapoint_size, replace=False)))

    start_locations = np.random.choice(list(range(datapoint_size - size_signal_area)), replace=False, size=number_start_locations)
    print(start_locations)

    ys_train = []
    ys_test = []
    for i in range(number_datapoints_train):
        func_idx = np.random.randint(0, funcs_per_class)
        func = functions_train[classes_train[i]][func_idx]
        y = func(x) + np.random.normal(0, noise_strength, datapoint_size)

        # start = np.random.randint(0, datapoint_size - size_signal_area)
        start_idx = np.random.randint(0, number_start_locations)
        start = start_locations[start_idx]
        y[start:start+size_signal_area] = signals[classes_train[i]]
        ys_train.append(y)

    for i in range(number_datapoints_test):
        func_idx = np.random.randint(0, funcs_per_class)
        func = functions_test[classes_test[i]][func_idx]
        y = func(x) + np.random.normal(0, noise_strength, datapoint_size)

        # start = np.random.randint(0, datapoint_size - size_signal_area)
        start_idx = np.random.randint(0, number_start_locations)
        start = start_locations[start_idx]
        y[start:start+size_signal_area] = signals[classes_test[i]]
        ys_test.append(y)

    train_dataset = utils.TensorDataset(torch.from_numpy(np.array(ys_train)).float(), torch.from_numpy(classes_train).long())
    test_dataset = utils.TensorDataset(torch.from_numpy(np.array(ys_test)).float(), torch.from_numpy(classes_test).long())

    train_dataloader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

class Squeeze(nn.Module):

    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Net(nn.Module):
    def __init__(self, vib, data_size, p_dropout, latent_size, num_classes):
        super(Net, self).__init__()
        self.vib = vib

        self.net = nn.Sequential(
            nn.Conv1d(1, 10, kernel_size=11),
            nn.ReLU(),
            Flatten(),
            nn.Linear((data_size-10)*10, latent_size),
            nn.Dropout(p_dropout),
            nn.ReLU(),
        )

        if vib:
            self.reg_layer = Bottleneck(latent_size, 256)
        else:
            self.reg_layer = nn.Linear(latent_size, 256)

        self.net2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        if self.vib:
            x, kl = self.reg_layer(x)
            kl = torch.sum(kl, dim=1) # TODO: Find out dimension!
        else:
            x = self.reg_layer(x)
            kl = torch.Tensor([0])
        x = self.net2(x).squeeze()
        return x, kl.mean()
