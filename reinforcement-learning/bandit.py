import numpy as np
import sys
import random
import math

np.random.seed(0)
arm_num = 10
arm_value = [0.0] * arm_num
arm_Q = []
times = 10000

def init():
    mu = 0
    sigma = 1
    for i in range(arm_num):
        value = np.random.normal(mu, sigma)
        arm_value[i] = value
        

def get_reward(index):
    """
    use normal distribution to return a reward
    """
    mu = arm_value[index]
    sigma = 1
    return np.random.normal(mu, sigma)


def greedy_alg():
    # init Q(a)
    arm_Q = [0.0] * arm_num
    count_Q = [0] * arm_num
    # greedy algorthm: always choose the current best option
    for count in range(times):
        # always choose the max Q
        select = arm_Q.index(max(arm_Q))
        # get reward
        reward = get_reward(select)
        # update Q(a): calculate average reward
        count_Q[select] = count_Q[select] + 1
        arm_Q[select] = arm_Q[select] + (reward - arm_Q[select]) / count_Q[select]
    # print 10 arms final reward
    print 'greedy algorithm result:'
    for e in arm_Q:
        sys.stdout.write('%.3f' % e + ' ')
    print ''


def e_greedy_alg():
    # init Q(a)
    arm_Q = [0.0] * arm_num
    count_Q = [0] * arm_num
    e = 0.1
    # e-greedy algorithm: 1-e probability to choose maximum index, otherwise random
    for count in range(times):
        # choose an arm
        r = random.uniform(0, 1)
        if r <= e:
            # random choose one arm
            select = random.randint(0, arm_num - 1)
        else:
            select = arm_Q.index(max(arm_Q))
        # get_reward
        reward = get_reward(select)
        # update Q(a)
        count_Q[select] = count_Q[select] + 1
        arm_Q[select] = arm_Q[select] + (reward - arm_Q[select]) / count_Q[select]
    # print 10 arms final reward
    print 'e-greedy algorithm result:'
    for e in arm_Q:
        sys.stdout.write('%.3f' % e + ' ')
    print ''
    print count_Q


def ucb_alg():
    """
    Upper confidence bound algorithm, introduce a term for confidence level.
    Often performs well but is not practical in reinforcement learning.
    Use At = argmax(Qt(a) + c * sqrt(lnt / Nt(a))) for action
    c * sqrt(lnt / Nt(a)) denotes confidence level
    """
    # init Q(a)
    arm_Q = [0.0] * arm_num
    count_Q = [0] * arm_num
    c = 0.1
    # ucb algorithms
    for count in range(times):
        # choose an arm
        evl = [0.0] * arm_num
        for i in range(arm_num):
            if count_Q[i] == 0:
                break
            else:
                evl[i] = arm_Q[i] + math.sqrt(c * math.log(count) / count_Q[i])
        if i == arm_num - 1 and count_Q[i] != 0:
            select = evl.index(max(evl))
        else:
            select = i
        # get_reward
        reward = get_reward(select)
        # update Q(a)
        count_Q[select] = count_Q[select] + 1
        arm_Q[select] = arm_Q[select] + (reward - arm_Q[select]) / count_Q[select]
    # print 10 arms final reward
    print 'e-greedy algorithm result:'
    for e in arm_Q:
        sys.stdout.write('%.3f' % e + ' ')
    print ''
    print count_Q


# init the 10 bandit
init()
print 'initial 10-arm bandits value'
for e in arm_value:
    sys.stdout.write('%.3f' % e + ' ')
print ''
# use greedy algorithm
greedy_alg()
# use e-greedy algorithm
e_greedy_alg()
# use ucb algorithm
ucb_alg()



        
    
