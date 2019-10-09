import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_rl
import gym
from bottleneck import Bottleneck

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(nn.Module):
    def __init__(self, obs_space, action_space, model_type="default", use_bottleneck=False,
                 dropout=0, use_l2a=False, use_bn=False, sni_type=None):
        super().__init__()

        # Decide which components are enabled
        self.use_bottleneck = use_bottleneck
        self.use_l2a = use_l2a
        self.dropout = dropout
        self.model_type = model_type
        self.sni_type = sni_type
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        print(n,m)

        # Define image embedding
        if model_type in ['large', 'semi-large']:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, (2, 2)),
                nn.ReLU(),
            )
            self.res1 = nn.Sequential(
                nn.Conv2d(32, 32, (3, 3), padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, (3, 3), padding=1),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 32, (2, 2)),
                nn.ReLU(),
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 32, (2, 2)),
                nn.ReLU(),
            )

            self.max_pool = nn.MaxPool2d((2,2))
            self.image_embedding_size = ((n-3)//2)*((m-3)//2)*32
            if model_type == 'large':
                self.res2 = nn.Sequential(
                    nn.Conv2d(32, 32, (3, 3), padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, (3, 3), padding=1),
                    nn.ReLU(),
                )
                self.res3 = nn.Sequential(
                    nn.Conv2d(32, 32, (3, 3), padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, (3, 3), padding=1),
                    nn.ReLU(),
                )
            # Max Pool

        elif model_type == "default":
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
            )
            self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
            assert not use_bn
        elif model_type == "default2":
            if use_bn:
                self.image_conv = nn.Sequential(
                    nn.Conv2d(3, 16, (2, 2)),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                    nn.BatchNorm2d(16),
                    nn.Conv2d(16, 32, (2, 2)),
                    nn.ReLU(),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 32, (2, 2)),
                    nn.ReLU(),
                    nn.BatchNorm2d(32),
                )
            else:
                self.image_conv = nn.Sequential(
                    nn.Conv2d(3, 16, (2, 2)),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                    nn.Conv2d(16, 32, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, (2, 2)),
                    nn.ReLU(),
                )
            self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*32
        elif model_type == "double_pooling":
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 16, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
            )
            self.image_embedding_size = (((n-1)//2-1)//2-1)*(((m-1)//2-1)//2-1)*64
            assert not use_bn
        print("Image embedding size: ", self.image_embedding_size)

        # Not supported with current bottleneck

        # Resize image embedding
        self.embedding_size = self.semi_memory_size

        # Define actor's model
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("Unknown action space: " + str(action_space))

        if use_bottleneck:
            assert self.dropout == 0
            self.reg_layer = Bottleneck(self.embedding_size, 64)
        else:
            self.reg_layer = nn.Linear(self.embedding_size, 64)

        self.dropout_layer = nn.Dropout(p=self.dropout, inplace=False)

        self.actor = nn.Sequential(
            # nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            # nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def encode(self, obs):
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        if self.model_type == 'large':
            x = self.conv1(x)

            x = self.res1(x) + x
            x = self.conv2(x)

            x = self.res2(x) + x
            x = self.conv3(x)

            x = self.res3(x) + x
            x = self.max_pool(x)

        elif self.model_type == 'semi-large':
            x = self.conv1(x)

            x = self.res1(x) + x
            x = self.conv2(x)
            x = self.conv3(x)

            x = self.max_pool(x)

        else:
            x = self.image_conv(x)
        embedding = x.reshape(x.shape[0], -1)

        if self.use_bottleneck:
            bot_mean, bot, kl = self.reg_layer(embedding)
            kl = torch.sum(kl, dim=1)
        elif self.use_l2a:
            bot_mean = bot = self.reg_layer(embedding)
            kl = torch.sum(bot**2, dim=1)
        else:
            bot_mean = self.reg_layer(embedding)
            bot = self.dropout_layer(bot_mean)
            kl = torch.Tensor([0])

        return bot_mean, bot, kl

    def compute_run(self, obs):
        bot_mean, bot, kl = self.encode(obs)

        if self.sni_type is not None:
            # For any SNI type, the rollouts values are deterministic
            x_dist = self.actor(bot_mean)
            dist = Categorical(logits=F.log_softmax(x_dist, dim=1))
            value = self.critic(bot_mean).squeeze(1)
        else:
            x_dist = self.actor(bot)
            dist = Categorical(logits=F.log_softmax(x_dist, dim=1))
            value = self.critic(bot).squeeze(1)
        return dist, value, kl

    def compute_train(self, obs):
        bot_mean, bot, kl = self.encode(obs)

        if self.sni_type == 'vib':
            # Need both policies for training, but still only one value function:
            x_dist_run = self.actor(bot_mean)
            dist_run = Categorical(logits=F.log_softmax(x_dist_run, dim=1))
            value = self.critic(bot_mean).squeeze(1)

            x_dist_train = self.actor(bot)
            dist_train = Categorical(logits=F.log_softmax(x_dist_train, dim=1))
            return dist_run, dist_train, value, kl
        elif self.sni_type == 'dropout' or self.sni_type is None:
            # Random policy AND value function
            x_dist_train = self.actor(bot)
            dist_train = Categorical(logits=F.log_softmax(x_dist_train, dim=1))
            value = self.critic(bot).squeeze(1)
            return dist_train, value, kl
