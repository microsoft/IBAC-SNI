import math
import operator
from functools import reduce

import numpy as np

import gym
from gym import error, spaces, utils

class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (env.agentPos, env.agentDir, action)

        # Get the count for this (s,a) pair
        preCnt = 0
        if tup in self.counts:
            preCnt = self.counts[tup]

        # Update the count for this (s,a) pair
        newCnt = preCnt + 1
        self.counts[tup] = newCnt

        bonus = 1 / math.sqrt(newCnt)

        reward += bonus

        return obs, reward, done, info

class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (env.agentPos)

        # Get the count for this key
        preCnt = 0
        if tup in self.counts:
            preCnt = self.counts[tup]

        # Update the count for this key
        newCnt = preCnt + 1
        self.counts[tup] = newCnt

        bonus = 1 / math.sqrt(newCnt)

        reward += bonus

        return obs, reward, done, info

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        # Hack to pass values to super wrapper
        self.__dict__.update(vars(env))
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']

class PartialObsFullGridWrapper(gym.core.ObservationWrapper):
    """
    Use same visual field as in the partial observable case, but embedded in the full grid
    """

    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

    def observation(self, obs):

        # Get a copy of the current grid just to be sure (as some functions have side effects)
        cur_grid = self.env.grid.copy()

        # Get area within vision (not taking occlusions into account)
        topX, topY, botX, botY = self.env.get_view_exts()

        # Get the Numpy array of the full grid
        full_grid = cur_grid.encode()

        # Encode the location of the agent + it's orientation 
        full_grid[self.env.agent_pos[0]][self.env.agent_pos[1]] = np.array([255, self.env.agent_dir, 0])
        
        # print(topX, botX)
        # print(topY, botY)
        topX, botX, topY, botY = list(np.clip([topX, botX, topY, botY], 0, [self.env.width]*2 + [self.env.height]*2))
        po_grid = np.zeros_like(full_grid)
        po_grid[topX:botX,topY:botY,:] = 1
        po_grid = full_grid * po_grid

        # print("------------")
        # print("Full Grid:", full_grid[:,:,0].T)
        # # print("Vis_mask:", vis_mask.astype(int))
        # print("Po_grid:", po_grid[:,:,0].T)
        # print("Direction:", self.env.agent_dir)
        # print("")

        obs = {
            'image': po_grid,
            'direction': self.env.agent_dir,
            'mission': self.env.mission
        }
        return obs

class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        self.__dict__.update(vars(env))  # hack to pass values to super wrapper
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width, self.env.height, 3),  # number of cells
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

    def observation(self, obs):
        full_grid = self.env.grid.encode()
        full_grid[self.env.agent_pos[0]][self.env.agent_pos[1]] = np.array([255, self.env.agent_dir, 0])
        
        obs = {
            'image': full_grid,
            'direction': self.env.agent_dir,
            'mission': self.env.mission
        }
        return obs

class FlatObsWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=64):
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, imgSize + self.numCharCodes * self.maxStrLen),
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, "mission string too long"
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs
