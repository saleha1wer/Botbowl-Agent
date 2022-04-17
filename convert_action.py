
from ast import Raise
from distutils import core
from turtle import position, right
from botbowl.ai.env import BotBowlEnv, EnvConf,RewardWrapper
from botbowl import Action, ActionChoice, ActionType,Square
import numpy as np 
import botbowl

import math

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def short_to_normal(env,game,idx):
    short_mask = np.load('short_mask.npy',allow_pickle=True)

    if idx < 20:
        # return env.env._compute_action_idx(Action(short_mask[idx]))
        return idx
    elif idx < 22: 
        # get action type 
        # return action with same action type
        # print('temp available actions: ',game.state.available_actions)        
        # a_type = env.env._compute_action(idx)[0].action_type
        # a_type = botbowl.ActionType.END_SETUP
        # return env.env._compute_action_idx(get_spacial_action(game,a_type))
        return idx
        # raise ValueError('idx is ', idx)
    else:
        # print('idx', idx)
        action = get_spacial_action(env,game,short_mask[idx])
        # print(action)
        if type(env) == BotBowlEnv:
            return env._compute_action_idx(action)
        else:
            return env.env._compute_action_idx(action)


def get_spacial_action(env,game,action_type):
    legal = game.state.available_actions
    correct = [action for action in legal if action.action_type == action_type]
    # print('correct',correct)
    if len(correct) == 0:
        raise ValueError('ilegal spacial move received')
    else:
        position = None
        player = None
        action_choice = np.random.choice(correct)
        if len(action_choice.players) == len(action_choice.positions) == 0: 
            action = Action(action_choice)

        elif len(action_choice.positions) > 0: # Select position
            side_width = game.arena.width / 2
            side_height = game.arena.height
            squares_from_left = math.ceil(side_width / 2)
            squares_from_right = math.ceil(side_width / 2)
            squares_from_top = math.floor(side_height / 2)
            left_center = Square(squares_from_left, squares_from_top)
            right_center = Square(game.arena.width - 1 - squares_from_right, squares_from_top)

            if action_choice.action_type == ActionType.PLACE_BALL: # Place ball 
                if game.is_team_side(left_center, game.active_team):
                    return Action(ActionType.PLACE_BALL, position=right_center)
                else:
                    return Action(ActionType.PLACE_BALL, position=left_center)

            else: 
                left = Square(1000,1)
                right = Square(-1000,1)
                for pos in action_choice.positions:
                    if pos.x < left.x:
                        left = pos
                    if pos.x > right.x:
                        right = pos
                # if env.env.home_team == game.get_turn_order()[0]:
                if game.is_team_side(left_center, game.active_team):
                    position = right
                else:
                    position = left

        if len(action_choice.players) > 0: # Select player
            player = np.random.choice(action_choice.players)

        action = Action(action_choice, position=position, player=player)

        return action


def normal_to_short_mask(normal_mask):
    short_mask = []
    for boo in normal_mask[:22]:
        short_mask.append(boo)
    spac_acts = chunks(normal_mask[22:],198)
    for act in spac_acts:
        if True in act:
            short_mask.append(True)
        else:
            short_mask.append(False)
    return short_mask
