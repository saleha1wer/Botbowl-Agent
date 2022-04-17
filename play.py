from email import policy
import enum
from re import L
import tensorflow as tf 
import botbowl.web.server as server
import math
import numpy as np
import botbowl
import tensorflow as tf
import sys
from botbowl.ai.env import BotBowlEnv, EnvConf,RewardWrapper
from reward_wrapper import A2C_Reward
from botbowl import Action, ActionChoice, ActionType,Square, BBDieResult, Skill, Formation, ProcBot
import time
from convert_action import short_to_normal,normal_to_short_mask

# new playing (make botbowl agent class)
# fix going to diff side sin diff halves

np.set_printoptions(threshold=sys.maxsize)

def temp():
    total_rew = 0

    for i in range(10):
        policy_network = tf.keras.models.load_model('saved_models/Network_Run699.tf')
        env_conf = EnvConf(size=5)
        env = BotBowlEnv(env_conf,away_agent='random')
        env = RewardWrapper(env, home_reward_func=A2C_Reward())
        done = False
        spatial_obs, non_spatial_obs, mask = env.reset()
        steps= 0
        env.reset() 
        spacial = []
        non_spacial = []
        win_rate = []
        turn = True
        while not done:
            print(env.game.active_team)
            if turn:
                env.render()
                spatial_obs, non_spatial_obs, mask = env.get_state()
                mask = normal_to_short_mask(mask)
                shape1 = spatial_obs.shape
                sample = [np.array(spatial_obs).reshape(1,shape1[0],shape1[1],shape1[2]),np.array([non_spatial_obs]).reshape(1,-1)]

                temp = policy_network.predict(sample)
                score = temp[0]
                action_dis = temp[1]
                action_dis = action_dis[0]
                for idx, v in enumerate(mask):
                    if v == False:
                        action_dis[idx] = -float('inf')

                act_idx = np.argmax(action_dis)

                act_idx = short_to_normal(env,env.game,act_idx)
                print(action_dis)
                print('action', act_idx)
                (temp, temp, temp), rew, done, temp = env.step(act_idx)


                steps += 1
                print('win_rate from PN', score)
                
                if abs(rew) > 0:
                    total_rew = total_rew + rew
                    print('reward',rew)

                print('home: ', env.game.state.home_team.state.score)
                print('away: ', env.game.state.away_team.state.score)

    print('total_reward per game', total_rew/10)



class MCTSbot(ProcBot):
    env: BotBowlEnv
    def __init__(self, name):
        super().__init__(name)
        env_conf = EnvConf(size=5)
        self.env = BotBowlEnv(env_conf,away_agent='human')

        self.my_team = None
        self.opp_team = None
        self.actions = []
        self.setup_actions = []
        self.last_turn = 0
        self.last_half = 0
        # env = RewardWrapper(env, home_reward_func=A2C_Reward())

    def new_game(self, game, team):
        self.my_team = team
        self.opp_team = game.get_opp_team(team)
        self.last_turn = 0
        self.last_half = 0

    def coin_toss_flip(self, game):
        """
        Select heads/tails and/or kick/receive
        """
        return Action(ActionType.TAILS)
    
    def coin_toss_kick_receive(self, game):
        """
        Select heads/tails and/or kick/receive
        """
        return self.network_action(game)

    def setup(self, game):
        """
        Use either a Wedge offensive formation or zone defensive formation.
        """
        # Update teams
        self.my_team = game.get_team_by_id(self.my_team.team_id)
        self.opp_team = game.get_opp_team(self.my_team)

        if self.setup_actions:
            action = self.setup_actions.pop(0)
            return action

        # If smaller variant - use built-in setup actions

        for action_choice in game.get_available_actions():
            if action_choice.action_type != ActionType.END_SETUP and action_choice.action_type != ActionType.PLACE_PLAYER:
                self.setup_actions.append(Action(ActionType.END_SETUP))
                return Action(action_choice.action_type)

        # This should never happen
        return None

    def perfect_defense(self, game):
        return self.network_action(game)

    def place_ball(self, game):
        """
        Place the ball when kicking.
        """
        side_width = game.arena.width / 2
        side_height = game.arena.height
        squares_from_left = math.ceil(side_width / 2)
        squares_from_right = math.ceil(side_width / 2)
        squares_from_top = math.floor(side_height / 2)
        left_center = Square(squares_from_left, squares_from_top)
        right_center = Square(game.arena.width - 1 - squares_from_right, squares_from_top)
        if game.is_team_side(left_center, self.opp_team):
            return Action(ActionType.PLACE_BALL, position=left_center)
        return Action(ActionType.PLACE_BALL, position=right_center)

    def high_kick(self, game):
        return self.network_action(game)

    def reroll(self, game):
        return self.network_action(game)
    
    def touchback(self, game):
        return self.network_action(game)

    def turn(self, game):
        return self.network_action(game)
    def use_pro(self, game):
        return self.network_action(game)

    def use_juggernaut(self, game):
        return self.network_action(game)

    def use_wrestle(self, game):
        return self.network_action(game)

    def use_stand_firm(self, game):
        return self.network_action(game)

    def use_bribe(self, game):
        return self.network_action(game)

    def quick_snap(self, game):
        return self.network_action(game)

    def blitz(self, game):
        return self.network_action(game)

    def player_action(self, game):
        return self.network_action(game)

    def block(self, game):
        return self.network_action(game)

    def push(self, game):
        return self.network_action(game)

    def follow_up(self, game):
        return self.network_action(game)

    def apothecary(self, game):
        return self.network_action(game)

    def interception(self, game):
        return self.network_action(game)

    def gfi(self, game):
        return self.network_action(game)

    def dodge(self, game):
        return self.network_action(game)

    def pickup(self, game):
        return self.network_action(game)

    def blood_lust_block_or_move(self, game):
        return self.network_action(game)

    def eat_thrall(self, game):
        return self.network_action(game)

    def network_action(self, game):
        # Select a random action type
        # if formation, set up and end setup 
        self.env.game = game
    
        spatial_obs, non_spatial_obs, mask = self.env.get_state() 
        mask = normal_to_short_mask(mask)

        shape1 = spatial_obs.shape
        sample = [np.array(spatial_obs).reshape(1,shape1[0],shape1[1],shape1[2]),np.array([non_spatial_obs]).reshape(1,-1)]
        policy_network = tf.keras.models.load_model('saved_models/Network_Run225.tf')
        temp = policy_network.predict(sample)
        score = temp[0]
        action_dis = temp[1]
        action_dis = action_dis[0]
        for idx, v in enumerate(mask):
            if v == False:
                action_dis[idx] = -float('inf')
            # if idx == 20 or idx == 21:
            #     action_dis[idx] = -float('inf')

        act_idx = np.argmax(action_dis)
        act_idx = short_to_normal(self.env,game,act_idx)
        action = self.env._compute_action(act_idx)

        action = action[0]
        print(action)
        return action

    def end_game(self, game):
        """
        Called when a game ends.
        """
        winner = game.get_winning_team()
        print("Casualties: ", game.num_casualties())
        if winner is None:
            print("It's a draw")
        elif winner == self.my_team:
            print("I ({}) won".format(self.name))
            print(self.my_team.state.score, "-", self.opp_team.state.score)
        else:
            print("I ({}) lost".format(self.name))
            print(self.my_team.state.score, "-", self.opp_team.state.score)

if __name__ == '__main__':

    botbowl.register_bot('mcts', MCTSbot)
    # Load configurations, rules, arena and teams
    # config = botbowl.load_config("web.json")
    # ruleset = botbowl.load_rule_set(config.ruleset)
    # arena = botbowl.load_arena(config.arena)
    # home = botbowl.load_team_by_filename("human", ruleset)
    # away = botbowl.load_team_by_filename("human", ruleset)
    # config.competition_mode = False
    # config.debug_mode = False

    server.start_server(debug=True, use_reloader=False, port= 1293)
    # Play 10 games
    # game_times = []
    # for i in range(10):
    #     away_agent = botbowl.make_bot("mcts")
    #     home_agent = botbowl.make_bot("my-random-bot")

    #     game = botbowl.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
    #     game.config.fast_mode = True

    #     print("Starting game", (i+1))
    #     game.init()
    #     print("Game is over")

