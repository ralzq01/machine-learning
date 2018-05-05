import gym
import random
import numpy
from gym.envs.classic_control import rendering

class GridMaze(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        # state space 5 * 5 grid
        self.state_space = [i for i in range(25)]
        self.trap_state = [3,5,9,12,16,23]
        self.goal_state = [24]

        # action space -go north, east, south, west
        self.action_space = ['n','e','s','w']

        # rewards
        self.reward_space = dict()
        # robot should not enter to a trap state
        for trap in self.trap_state:
            self.reward_space[trap] = -40
        # robot will get bonus on specific state
        for bonus in self.goal_state:
            self.reward_space[bonus] = bonus
        # every step will consume 1 bonus
        
        # initial state
        self.state = None

        # visualize
        self.viewer = None
    
    def move(self, action):
        """
        move base on the current state
        """
        # next state
        if action == 'n':
            if self.state > 4:
                self.state -= 5
        elif action == 's':
            if self.state < 20:
                self.state += 5
        elif action == 'e':
            if self.state % 5 != 4:
                self.state += 1
        else:
            if self.state % 5 != 0:
                self.state -= 1
        

    def step(self, action):
        """
        take a action on current state
        return observation, reward(for current state), done, info
        """

        # terminate state
        if self.state in self.goal_state + self.trap_state:
            return None, self.reward_space[self.state], True, {}
        
        # reward
        if self.state in self.reward_space:
            reward = self.reward_space[self.state] - 1
        else:
            reward = -1
        
        # take the action
        self.move(action)

        return self.state, reward, False, {}

    def reset(self):
        # random initialize start position
        self.state = self.state_space[int(random.random() * len(self.state_space))]
        while self.state in self.trap_state:
            self.state = self.state_space[int(random.random() * len(self.state_space))]
        return self.state

    def render(self, mode='human', close=False):
        """
        show the senerio on the screen
        5 * 5 grid with 60 * 60
        """

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        # parameter for visualizing
        length = 60
        row = 5
        col = 5
        # set screen size
        screen_width = length * col
        screen_height = length * row

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # row line
            rowline = [rendering.Line((0, i * length), (col * length, i * length)) for i in range(row + 1)]
            # col line
            colline = [rendering.Line((i * length, 0), (i * length, row * length)) for i in range(col + 1)]
            # add to geom
            for line in rowline + colline:
                self.viewer.add_geom(line)
            # create trap and bonus
            for state in self.trap_state + self.goal_state:
                circle = rendering.make_circle(25)
                circle_loc = rendering.Transform(
                    translation=((state % 5) * length + length / 2, state / 5 * length + length / 2)
                )
                circle.add_attr(circle_loc)
                if state in self.trap_state:  
                    circle.set_color(0,0,0)
                else:
                    circle.set_color(0,0.6,0.4)
                self.viewer.add_geom(circle)
            # create agent
            self.agent = rendering.make_circle(25)
            self.agent_loc = rendering.Transform()
            self.agent.add_attr(self.agent_loc)
            self.agent.set_color(0.8,0.6,0.4)
            self.viewer.add_geom(self.agent)

        # show agent location
        if self.state is None:
            return None
        self.agent_loc.set_translation(
            (self.state % 5) * length + length / 2,
            (self.state / 5) * length + length / 2
        )

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
    

    def get_action_space(self):
        if self.state is None:
            return self.action_space
        actions = []
        if self.state % 5 != 0:
            actions += ['w']
        if self.state % 5 != 4:
            actions += ['e']
        if self.state / 5 != 0:
            actions += ['n']
        if self.state / 5 != 4:
            actions += ['s']
        return actions
    
