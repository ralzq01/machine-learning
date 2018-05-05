import sys
import gym
from qlearning import *
import time
from gym import wrappers

# parameter defination
epsilon = 0.2
alpha = 0.1
gamma = 0.8
episode = 1000

if __name__ == "__main__":
    
    # training
    print 'start training...'
    qfunc = qlearning(
        episode_num=episode,
        alpha=alpha,
        epsilon= epsilon,
        gamma=gamma
    )

    # result
    """print 'training result:'
    for key in qfunc:
        print key, qfunc[key]"""
    
    # testing
    print 'start testing...'
    for i in range(10):
        # initialize environment
        s = env.reset()
        env.render()
        time.sleep(1)
        # start exploitation
        terminate = False
        while not terminate:
            a = greedy(qfunc, s)
            s, _, terminate, _ = env.step(a)
            env.render()
            time.sleep(1)

        print '[test] episode %d is finished' % i
