
import numpy as np
from botbowl.ai.env import BotBowlEnv, EnvConf,RewardWrapper, deepcopy
from network import Network
from reward_wrapper import A2C_Reward
from botbowl.core.procedure import *
import sys
from replay_buffer import Buffer
from convert_action import short_to_normal, normal_to_short_mask 
import math
from node import Node

class MCTS_Run():
    def __init__(self,network,max_steps=300):
        self.max_steps = max_steps
        self.network = network

    def select(self,env,node,turn):
        # selects the node with the highest UCB until a leaf node is reached
        current = node
        
        spacials = []
        non_spacials = []
        turns = []
        distributions = []
        spatial_obs, non_spatial_obs, action_mask = node.spacials, node.non_spacials, node.action_mask
        temp = self.network.network.predict([np.array(spatial_obs).reshape(1,44,11,18),np.array(non_spatial_obs).reshape(1,-1)])
        act_dis = temp[1][0]


        if not node.is_expanded:
            self.expand(env,current)

        ucb_values = [i.ucb() for i in current.children]
        # ucb_values = ucb_values[0]
        max_idx = np.argmax(ucb_values)
        children_idx = [i.action for i in current.children]

        # act_idx = current.children[max_idx].action 
        temp = np.zeros(act_dis.shape)
        for idx,child_idx in enumerate(children_idx):
            temp[child_idx] = ucb_values[idx]
            current.children[idx].prior = act_dis[child_idx]
        temp = temp/np.linalg.norm(temp)

        act_dis = [max(0,i) for i in temp]
        spacials.append(spatial_obs),non_spacials.append(non_spatial_obs),distributions.append(act_dis),turns.append(turn)

        while len(current.children) > 0:
            print(len(current.children))
            ucb_values = [i.ucb() for i in current.children]
            max_idx = np.argmax(ucb_values)
            if len(current.children) == 1:
                current = current.children[0]
            else:
                current = current.children[max_idx]


            turn = 1 if current.game.game.active_team == current.game.env.home_team else -1
            spatial_obs, non_spatial_obs, action_mask = current.spacials, current.non_spacials, current.action_mask
            temp = self.network.network.predict([np.array(spatial_obs).reshape(1,44,11,18),np.array(non_spatial_obs).reshape(1,-1)])
            act_dis = temp[1][0]
            if len(current.children) > 0:
                ucb_values = [i.ucb() for i in current.children]
                if not isinstance(ucb_values[0],float):
                    ucb_values = ucb_values[0]

                max_idx = np.argmax(ucb_values)
                children_idx = [i.action for i in current.children]
                if not isinstance(children_idx,list):
                    children_idx = children_idx[0]
                # act_idx = current.children[max_idx].action 
                temp = np.zeros(act_dis.shape)
                for idx,child_idx in enumerate(children_idx):
                    temp[child_idx] = ucb_values[idx]
                    current.children[idx].prior = act_dis[child_idx]
                temp = temp/np.linalg.norm(temp)
                act_dis = [max(0,i) for i in temp]

                spacials.append(spatial_obs),non_spacials.append(non_spatial_obs),distributions.append(act_dis),turns.append(turn)

                # reward = self.expand(current)
                # if reward is not None:
                #     break
            
        turns = np.array(turns)
        return current,spacials,non_spacials,distributions,turns

    def expand(self,env,node):
        if node.is_expanded:
            raise ValueError('node already expanded')
        reward = None
        proc = node.game.game.get_procedure()
        game_copy = node.game
        root_step = game_copy.game.get_step()
        if isinstance(proc, CoinTossFlip):
            mask = node.action_mask
            i = [i for i in range(39) if mask[i] == True]
            i = np.random.choice(i)
            normal_idx = short_to_normal(game_copy,game_copy.game,i)
            (temp, temp, temp), reward, done, temp  = game_copy.step(normal_idx)
            if not done:
                spatial_obs, non_spatial_obs, action_mask = game_copy.get_state()
                obs = [spatial_obs, non_spatial_obs, action_mask]
                child = Node(deepcopy(game_copy),obs,1,action=i,parent=node)
                node.children.append(child)
            game_copy.game.revert(root_step)
        elif isinstance(proc, Setup):
            temp, temp, real_mask = game_copy.get_state()
            i = [z for z in range(len(real_mask)) if real_mask[z] == True]
            idx = np.random.choice(i)
            (temp, temp, temp), reward, done, temp  = game_copy.step(idx)
            if not done:
                spatial_obs, non_spatial_obs, action_mask = game_copy.get_state()
                obs = [spatial_obs, non_spatial_obs, action_mask]
                child = Node(deepcopy(game_copy),obs,1,action=idx,parent=node)
                node.children.append(child)
            game_copy.game.revert(root_step)

        elif isinstance(proc, PlaceBall):
            temp, temp, real_mask = game_copy.get_state()
            i = [z for z in range(len(real_mask)) if real_mask[z] == True]

            side_width = game_copy.game.arena.width / 2
            side_height = game_copy.game.arena.height
            squares_from_left = math.ceil(side_width / 2)
            squares_from_right = math.ceil(side_width / 2)
            squares_from_top = math.floor(side_height / 2)
            left_center = Square(squares_from_left, squares_from_top)
            right_center = Square(game_copy.game.arena.width - 1 - squares_from_right, squares_from_top)
            if game_copy.game.is_team_side(left_center, game_copy.game.active_team):
                action= Action(ActionType.PLACE_BALL, position=right_center)
            else:
                action= Action(ActionType.PLACE_BALL, position=left_center)
            idx = game_copy.env._compute_action_idx(action)

            (temp, temp, temp), reward, done, temp  = game_copy.step(idx)

            if not done:
                spatial_obs, non_spatial_obs, action_mask = game_copy.get_state()
                obs = [spatial_obs, non_spatial_obs, action_mask]
                child = Node(deepcopy(game_copy),obs,1,action=22,parent=node)
                node.children.append(child)
            game_copy.game.revert(root_step)
        else:
            for i in range(39):
                game_copy = node.game
                root_step = game_copy.game.get_step()
                mask = node.action_mask
                temp = self.evaluate_game(game_copy)
                act_dis_root = temp[1]
                act_dis_root = act_dis_root[0]
                if mask[i] == True:
                    # convert i to idx real value (0-3388)
                    # print('short', i)
                    normal_idx = short_to_normal(game_copy,game_copy.game,i)
                    # print('normal', normal_idx)
                    # print('expanding node: available actions: ', game_copy.game.state.available_actions)
                    (temp, temp, temp), reward, done, temp  = game_copy.step(normal_idx)
                    if not done:
                        spatial_obs, non_spatial_obs, action_mask = game_copy.get_state()
                        obs = [spatial_obs, non_spatial_obs, action_mask]
                        sel_prob = act_dis_root[i]
                        child = Node(deepcopy(game_copy),obs,sel_prob,action=i,parent=node)
                        node.children.append(child)
                    game_copy.game.revert(root_step)
        node.is_expanded = True
        return reward
        # elif mask[i] == False:

        #     child = Node(game_copy,action=i,parent=node)
        #     child.evaluations.append(-10)
        #     node.children.append(child)

    def evaluate_node(self,node):
        spatial_obs, non_spatial_obs, mask = node.state.get_state() 

        sample = np.array([np.array(spatial_obs,),np.array([non_spatial_obs,])])

        temp = self.network.network.predict(sample)
        score = temp[0]
        action_dis = temp[1]
        return score,action_dis

    def evaluate_game(self,game):
        spatial_obs, non_spatial_obs, mask = game.get_state() 
        shape1 = spatial_obs.shape
        sample = [np.array(spatial_obs).reshape(1,shape1[0],shape1[1],shape1[2]),np.array([non_spatial_obs]).reshape(1,-1)]

        temp = self.network.network.predict(sample)
        score = temp[0]
        action_dis = temp[1]
        return score,action_dis

    def back_up(self,node,score,turn):
        current = node
        while current.parent is not None:
            #check turn (if player 2 --> v -= score)
            turn = 1 if current.game.game.active_team == current.game.env.home_team else -1
            current.visit(score*turn)
            current = current.parent

    def simulate(self,env,node,turn,reward_goal):
        # simulate using network
        # store spacials, non spacials, and action dis of every state 
        # calc score when done and return array, win_rate, of same length as the other arrays containing the score and -score 
        done = False
        steps = 0
        # spacials = []
        # non_spacials = []
        # turns = []
        # distributions = []
        game_copy = deepcopy(node.game)
        total_reward = 0
        prints = []
        while not done and steps < self.max_steps:

            spatial_obs, non_spatial_obs, action_mask = game_copy.get_state() 
            score, a_dis = self.evaluate_game(game_copy)
            a_dis = a_dis[0]
            noise = np.random.normal(0,0.1,a_dis.shape[0])
            a_dis = a_dis + noise
            norm = np.linalg.norm(a_dis)
            a_dis = a_dis/norm
            action_mask = normal_to_short_mask(action_mask)
            # print('action mask ', action_mask)
            # print('simulate, available actions: ', [action.action_type for action in game_copy.game.state.available_actions])
            for idx, action in enumerate(a_dis):
                if action_mask[idx] == False:
                    a_dis[idx] = -1
                if idx == 25: #
                    a_dis[idx] = -1

                # if idx == 20 or idx == 21:
                #     a_dis[idx] = -1

            action_idx = np.argmax(a_dis)
            prints.append(action_idx)
            action_idx = short_to_normal(game_copy,game_copy.game,action_idx)
            # print(a_dis)
            (temp, temp, temp), reward, done, temp = game_copy.step(action_idx)
            # temp = np.zeros(a_dis.shape)
            # temp[action_idx] = 1
            # a_dis = temp
            # if steps != 0:
            #     spacials.append(spatial_obs)
            #     non_spacials.append(non_spatial_obs)
            #     distributions.append(a_dis)
            #     turns.append(turn)

            steps += 1
            # total_reward += -turn * reward
            turn = 1 if game_copy.game.active_team == game_copy.env.home_team else -1
            if abs(reward) > reward_goal: 
                total_reward = -turn * reward
                done = True
        
        # spacials = np.array(spacials)
        # non_spacials = np.array(non_spacials)
        # turns = np.array(turns)
        print('simulation actions -->',prints)
        # distributions = np.array(distributions)
        print('reward times turn: ',total_reward)
        print('reward from game: ',reward)
       
        # return total_reward,spacials,non_spacials,turns,distributions
        return total_reward


    def one_run(self,reward_goal,root=None):
        if root == None:
            env_conf = EnvConf(size=5)
            env = BotBowlEnv(env_conf,away_agent='human') 
            env.reset()
            # env.render(feature_layers=True)
            env = RewardWrapper(env, home_reward_func=A2C_Reward())
            env.game.enable_forward_model()
            spatial_obs, non_spatial_obs, action_mask = env.get_state() 
            obs = (spatial_obs, non_spatial_obs, action_mask)
            root = Node(env,obs,1,env=env)
        else:
            root.env.reset()
            env = root.env
        # else:
        #     env_conf = EnvConf(size=5)
        #     env = BotBowlEnv(env_conf) 
        #     env.reset()
        #     # env.render(feature_layers=True)
        #     env = RewardWrapper(env, home_reward_func=A2C_Reward())
        #     env.game.enable_forward_model()
        #     root.state = env
            
        # Selection
        print('root: ',root)
        
        turn = 1 if env.game.active_team == env.env.home_team else -1

        temp = self.select(env,root,1)

        node = temp[0]

        sel_spacials,sel_non_spacials,sel_distributions,sel_turns = temp[1],temp[2],temp[3],temp[4]
        temp = node
        selected_actions = []

        while temp.parent is not None:
            selected_actions.append(temp.action)
            temp = temp.parent

        print('selected node: ',list(reversed(selected_actions)))
        # Expansion
        if not node.is_expanded: 
            reward = self.expand(env,node)


        # Back up if terminal 
        # make this return data as well
        if len(node.game.game.get_available_actions()) == 0: 
            print('Terminal node selected')
            winner = node.game.env.get_winner()
            if node.game.is_home_team(winner):
                score = 1
            elif winner == None:
                score = 0
            else: 
                score = -1

            self.back_up(node,score)
            return root, sel_spacials,sel_non_spacials,sel_turns*score,sel_distributions
        elif reward != 0 and reward != None:
            self.back_up(node,reward,sel_turns[-1])
            score = reward
        else:
        # Simulation
            # score,spacials,non_spacials,turns,distributions = self.simulate(node,sel_turns[-1],reward_goal)
            score = self.simulate(env,node,sel_turns[-1],reward_goal)

        # Back up 
            self.back_up(node,score,sel_turns[-1])
            # spacials = np.concatenate((np.array(sel_spacials),np.array(spacials)))
            # non_spacials = np.concatenate((np.array(sel_non_spacials),np.array(non_spacials)))
            # turns = np.concatenate((np.array(sel_turns),np.array(turns)))
            # distributions = np.concatenate((np.array(sel_distributions),np.array(distributions)))

            # return root,spacials,non_spacials,turns*score,distributions
            sel_spacials,sel_non_spacials,sel_distributions = np.array(sel_spacials),np.array(sel_non_spacials),np.array(sel_distributions)
            print('simulation over. win_rates returned')
            print(sel_turns*score)

        return root,sel_spacials,sel_non_spacials,sel_turns*score,sel_distributions


    def run_mcts(self, num_runs):
        sample_size = 50
        replay_memory = Buffer(100)
        wr_per_run = []
        print('Run 0')
        reward_goal = 0
        root,spacials,non_spacials,win_rates,distributions = self.one_run(reward_goal)
        replay_memory.add_to_buffer((np.array(spacials),non_spacials,distributions,win_rates))
        # self.network.train_network(spacials, non_spacials, win_rates,distributions,5)
        if len(win_rates) > 0:
            print('Reward: ', win_rates[0])
        for run in range(1,num_runs):
            print('Run {}'.format(run))
            for i in range(3):
                root,spacials,non_spacials,win_rates,distributions = self.one_run(reward_goal,root=root)
                spacials, non_spacials, win_rates, distributions = np.array(spacials),np.array(non_spacials),np.array(win_rates),np.array(distributions)
                replay_memory.add_to_buffer((spacials,non_spacials,distributions,win_rates))
                try: 
                    wr_per_run.append(abs(win_rates[0]))
                except:
                    print('none')
            # self.network.train_network(spacials, non_spacials, win_rates,distributions,5)
            if run in range(0,5000,25):
                self.network.save_network('Network_Run{}'.format(run))
            # if run in range(0,5000,50):
            #     pass
                # play against random bot
            print('Reward: ', win_rates[0])

            spacials,non_spacials,distributions,win_rates = replay_memory.get_from_buffer(sample_size)
            print('training win_rates')
            print(win_rates)
            self.network.train_network(spacials, non_spacials, win_rates,distributions,10)
            if run == 200:
                reward_goal = 0.1
                replay_memory.set_len(150)
            if run == 300:
                reward_goal = 0.2
            if run == 400:
                reward_goal = 0.4
            if run == 500:
                reward_goal = 0.9
                
            if run == 100:
                replay_memory.set_len(200)
            if run == 250:
                replay_memory.set_len(500)
            print('root number of children', len(root.children))

        self.network.save_network('Network_Run{}'.format(run))
        return wr_per_run
