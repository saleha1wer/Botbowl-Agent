

import numpy as np
from botbowl.ai.env import BotBowlEnv, EnvConf
from network import Network
import tensorflow as tf
import matplotlib.pyplot as plt

# reward

# rewards = np.load('rewards.npy',allow_pickle=True)
# values = []
# for r in range(int(len(rewards)/3)):
#     val = (rewards[r] + rewards[r+1]+ rewards[r+2])/ 3
#     values.append(val)

# X = range(len(values))
# plt.plot(X,values)
# plt.ylabel('Average Reward')
# plt.xlabel('Monte Carlo Tree Search Training Episode')
# plt.savefig('average-reward-per-MTE')
# plt.show()
# X = range(len(rewards))
# plt.plot(X,rewards)
# plt.savefig('reward-per-MCTS-simul')
# plt.show()


# VS random

scored = np.load('random_scored.npy',allow_pickle=True)
conceded = np.load('random_conceded.npy',allow_pickle=True)


X = range(len(scored))
plt.plot(X,scored,label='Scored')
plt.plot(X,conceded,label='Conceded')
plt.legend()
plt.ylabel('Number of touchdowns')
plt.xlabel('Games')
# plt.savefig('score-random')
plt.show()

# VS scripted

# scored = np.load('scripted_scored.npy',allow_pickle=True)
# conceded = np.load('scripted_conceded.npy',allow_pickle=True)

# X = range(len(scored))
# plt.plot(X,scored,label='Scored')
# plt.plot(X,conceded,label='Conceded')
# plt.ylabel('Number of touchdowns')
# plt.xlabel('Games')
# plt.savefig('score-scripted')
# plt.show()
