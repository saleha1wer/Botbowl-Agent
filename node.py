
import numpy as np
from botbowl.ai.env import BotBowlEnv, EnvConf,RewardWrapper, deepcopy
from network import Network
from reward_wrapper import A2C_Reward
import sys
from replay_buffer import Buffer
import botbowl
np.set_printoptions(threshold=sys.maxsize)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class Node: 
    # from https://github.com/njustesen/botbowl/blob/main/docs/search-based.md
    def __init__(self,game, objects,prior,action=None, parent=None,env=None):
        self.num_visits = 10**(-30)
        self.evaluations = [10**(-30)]
        self.game = game
        self.spacials = objects[0]
        self.non_spacials = objects[1]
        temp = objects[2]
        mask = []
        for boo in temp[:22]:
            mask.append(boo)
        spac_acts = chunks(temp[22:],198)
        for act in spac_acts:
            if True in act:
                mask.append(True)
            else:
                mask.append(False)

        self.action_mask = mask
        self.prior = prior
        self.action = action
        self.parent = parent
        self.is_expanded = False
        self.children = []
        self.env = env
        
    def visit(self, score):
        self.evaluations.append(score)
        self.num_visits += 1

    def ave_eval(self):
        return np.average(self.evaluations)
    
    def ucb(self):
        c = 1
        Q = self.ave_eval()
        ln_this = np.log(self.parent.num_visits)/ self.num_visits

        return Q + (c * self.prior * np.sqrt(ln_this))



















# botbowl.register_bot('my-random-bot', MyRandomBot)
# # Load configurations, rules, arena and teams
# config = botbowl.load_config("gym-5.json")
# ruleset = botbowl.load_rule_set(config.ruleset)
# arena = botbowl.load_arena(config.arena)
# home = botbowl.load_team_by_filename("human", ruleset)
# away = botbowl.load_team_by_filename("human", ruleset)
# config.competition_mode = False
# config.debug_mode = False

# # Play 10 games
# game_times = []
# for i in range(10):
#     away_agent = botbowl.make_bot("my-random-bot")
#     home_agent = botbowl.make_bot("my-random-bot")

#     game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
#     game.config.fast_mode = True

#     print("Starting game", (i+1))
#     game.init()
#     print("Game is over")

