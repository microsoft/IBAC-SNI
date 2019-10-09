from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class ChoiceEnv(MiniGridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size, gap, max_steps, doors, neg_reward, pos_reward, random_box_loc_size, sp_range, rotate_grid,
                 rand_x_start, goal_pos, num_colors):
        self.gap = gap
        self.neg_reward = neg_reward
        self.pos_reward = pos_reward
        self.doors = doors
        self.random_box_loc_size = random_box_loc_size
        self.sp_range = sp_range
        self.rotate_grid = rotate_grid
        self.rand_x_start = rand_x_start
        self.goal_pos = goal_pos
        self.num_colors = num_colors
        super().__init__(
            grid_size=size,
            max_steps=max_steps, # 10 * size * size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        # self.grid.set(width - 2, height - 2, Goal())
        self.grid.set(self.goal_pos[0], self.goal_pos[1], Goal())

        # Create a vertical splitting wall
        splitIdx = self._rand_int(3, width-2)
        self.grid.vert_wall(splitIdx, self.gap + 1)

        # Place a door in the wall
        # doorIdx = self._rand_int(1, height-2)
        colors = self.np_random.choice(list(COLORS.keys())[:self.num_colors], replace=False, size=self.doors)
        correct_clr_idx = self._rand_int(0,self.doors)
        door_rewards = [self.neg_reward] * self.doors
        door_rewards[correct_clr_idx] = self.pos_reward
        for i in range(self.doors):
            self.grid.set(splitIdx, height-2-i, Door(colors[i], is_locked=False, reward=door_rewards[i]))

        x_start_size = 1
        if self.rand_x_start:
            # x_start_size = self._rand_int(1, splitIdx)
            x_start_size = self._rand_int(1, width-1)
        # Reset starting position, change later if rotating
        self.start_pos = self.place_agent(top=(1, height-1-self.sp_range), size=(x_start_size, self.sp_range))
        self.place_obj(
            obj=Box(colors[correct_clr_idx]),
            top=(1, 1),
            size=self.random_box_loc_size
        )

        if self.rotate_grid:
            assert not self.rand_x_start
            nr_rotations = self._rand_int(0, 4)
            for i in range(nr_rotations):
                self.grid = self.grid.rotate_left()
            if nr_rotations == 0:
                self.start_pos = self.place_agent(top=(1, height-2), size=(1, 1))
            elif nr_rotations == 1:
                self.start_pos = self.place_agent(top=(width-2, height-2), size=(1, 1))
            elif nr_rotations == 2:
                self.start_pos = self.place_agent(top=(width-2, 1), size=(1, 1))
            elif nr_rotations == 3:
                self.start_pos = self.place_agent(top=(1, 1), size=(1, 1))
        # Place the agent in the bottom left corner

        self.mission = "Go to the goal as quickly as possible. The door of equal color to the box has a small positive reward to open, the others a small negative reward"


register(
    id='MiniGrid-Choice-9x9-v0',
    entry_point='gym_minigrid.envs:ChoiceEnv',
    reward_threshold=1000.0,
    kwargs = {
        'size': 9,
        'gap': 3,
        'max_steps': 10 * 9 * 9,
        'doors': 2,
        'neg_reward': -2.5,
        'pos_reward': 1.,
        'random_box_loc_size': (1, 1),
        'sp_range': 1,
        'rotate_grid': False,
        'rand_x_start': False,
        'goal_pos': (7, 7),
        'num_colors': 4,
    }
)

register(
    id='MiniGrid-Choice-11x11-v0',
    entry_point='gym_minigrid.envs:ChoiceEnv',
    reward_threshold=1000.0,
    kwargs = {
        'size': 11,
        'gap': 2,
        'max_steps': 10 * 9 * 9,
        'doors': 3,
        'neg_reward': -.4,
        'pos_reward': .6,
        'random_box_loc_size': (1, 1),
        'sp_range': 3,
        'rotate_grid': False,
        'rand_x_start': False,
        'goal_pos': (9, 5)
    }
)
