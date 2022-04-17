
from MCTS import MCTS_Run
from network import Network
import numpy as np 


num_runs = 700
shape_x1 = (44, 11, 18)
shape_x2 = (113)
actdis_len = 39
network = Network(shape_x1,shape_x2,actdis_len)
network.initialize_network()
mcts = MCTS_Run(network)
rewards = mcts.run_mcts(num_runs)

np.save('rewards',np.array(rewards))


# Run 1000 MCTS runs, every 10 runs test against random bot and store win rate.
# Store total reward per mcts run
# Save network every 50 runs

