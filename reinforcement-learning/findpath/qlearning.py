import gym
import environment
import time
import random

env = gym.make('maze-v0')

def greedy(qfunc, state):
    """
    choose a action which will result a best reward
    param: qfunc: a dictionary, key type: (state, action)
    return: action
    """
    action_space = env.get_action_space()
    if (state, action_space[0]) not in qfunc:
        qfunc[(state, action_space[0])] = 0
    qmax = qfunc[(state, action_space[0])]
    best_action = action_space[0]
    for action in action_space:
        if (state, action) not in qfunc:
            qfunc[(state, action)] = 0.0
        elif qfunc[(state, action)] > qmax:
            qmax = qfunc[(state, action)]
            best_action = action
    return best_action


def epsilon_greedy(qfunc, state, epsilon):
    """
    choose a action which will result a best reward
    based on posibility of 1 - epsilon
    otherwise (posibility of epsilon) choose another random action
    """

    greedy_action = greedy(qfunc, state)
    actions = env.get_action_space()
    # get every action taken posibility
    distribute = [epsilon / len(actions) for _ in range(len(actions))]
    distribute[actions.index(greedy_action)] += 1 - epsilon

    # choose
    r = random.random()
    total = 0.0
    for i in range(len(actions)):
        total += distribute[i]
        if total > r:
            return actions[i]
    return actions[len(actions) - 1]


def qlearning(episode_num, alpha, epsilon, gamma):
    """
    use qlearning for updating Q(s,a)
    paramter:
        episode_num: number of episode will be taken into training
        alpha: learning rate
        epsilon: for e-greedy usage
        gamma: discount factor
    return: q(s,a)
    """

    # initialize q(s,a)
    qfunc = dict()
    
    for iter in range(episode_num):
        # initialize senerio
        s = env.reset()
        terminate = False
        # shreshould make sure the episode won't execute forever
        threshould = 0
        # start episode
        while terminate == False and threshould < 100:
            # use e-greedy policy to choose an action
            a = epsilon_greedy(qfunc, s, epsilon)
            # observe the env
            next_s, reward, terminate, _ = env.step(a)
            # choose a best action for next_s
            amax = greedy(qfunc, next_s)
            # update Q(s, a)
            error = reward + gamma * qfunc[(next_s, amax)] - qfunc[(s, a)]
            qfunc[(s, a)] = qfunc[(s, a)] + alpha * error
            # update loop parameter
            s = next_s
            threshould += 1

    return qfunc

