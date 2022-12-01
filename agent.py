import numpy as np
import utils

## SHOULD I BE NOT UPDATING WITH Q_MAX IF DEAD?

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()     
 
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
   
    def find_best_action(self, s_prime):
        # 1. Choose optimal action based on Q-value or lack of exploration
            # action = argmax( f(Q(s,a), N(s,a)) )
                # f( Q(s,a), N(s,a) ) = 1 if N(s,a) < N_e
                # else                = Q(s,a)        
        best_action = 0
        q_val = -999
        if self._train: # WITH EXPLORATION
            for action in self.actions:
                if self.N[s_prime][action] < self.Ne:
                    tmp = 1
                    if tmp >= q_val:
                        q_val = tmp
                        best_action = action
                else:
                    tmp = self.Q[s_prime][action]
                    if tmp >= q_val:
                        q_val = tmp
                        best_action = action

        else: # WITHOUT EXPLORATION
            for action in self.actions:
                tmp = self.Q[s_prime][action]
                if tmp >= q_val:
                    q_val = tmp
                    best_action = action

        return best_action
    
    def calculate_reward(self, points, dead):
        # 2. From the result of the action on the environment, the agent obtains a reward r_t
        r_t = -.1
        if dead:
            r_t = -1
        elif points > self.points: # if food pellet found
            r_t = 1
        return r_t

    def update_q(self, r_t, s_prime):
        alpha = self.C / (self.C + self.N[self.s][self.a])
        if r_t == -1: # if dead, assume no next move
            self.Q[self.s][self.a] += alpha * (r_t - self.Q[self.s][self.a])
        else:
            maxQ = self.get_max_q(s_prime)
            self.Q[self.s][self.a] += alpha * (r_t + self.gamma * maxQ - self.Q[self.s][self.a] )

    def get_max_q(self, s_prime):
        argmax_list = [0,0,0,0]
        for action in self.actions:
            argmax_list[action] = self.Q[s_prime][action]
        
        return max(argmax_list)


    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''        
        s_prime = self.generate_state(environment)
        
        if self._train and self.a != None and self.s != None:
            r_t = self.calculate_reward(points, dead)
            self.N[self.s][self.a] += 1
            self.update_q(r_t, s_prime)
        
        if not dead:
            self.s = s_prime
            self.points = points
        else:
            self.reset()
            return 0

        # 1. Choose optimal action based on Q-value (or lack of exploration)
        a_prime = self.find_best_action(s_prime)
        if self._train and self.a != None and self.s != None: self.N[s_prime][a_prime] += 1
        self.a = a_prime
        return a_prime


    # Helper function to generate a state given an environment 
    # Each state in the MDP is a tuple of the form returned
    def generate_state(self, environment):
        hx, hy, body, fx, fy = environment
        
        fdx = 0
        if hx > fx:
            fdx = 1 # food on snake head left
        elif hx < fx:
            fdx = 2 # food on snake head right
        
        fdy = 0
        if hy > fy:
            fdy = 1 # food on snake head top
        elif hy < fy:
            fdy = 2 # food on snake head bottom

        awx = 0
        if hx == 1:
            awx = 1 # wall on snake head left
        if hx == utils.DISPLAY_WIDTH - 2: # -2 b/c of 0-indexing
            awx = 2 # wall on snake head right
        
        awy = 0
        if hy == 1:
            awy = 1 # wall on snake head top
        if hy == utils.DISPLAY_HEIGHT - 2: # -2 b/c of 0-indexing
            awx = 2 # wall on snake head bottom

        # Check where the snake's body is in relation to it's head
        adtop = 0
        adbot = 0
        adleft = 0
        adright = 0
        for i in body:
            if i[1] - hy == -1 and i[0] == hx:
                adtop = 1 # adjoining top square has snake body
            if i[1] - hy == 1 and i[0] == hx:
                adbot = 1 # adjoining bottom square has snake body
            if i[0] - hx == -1 and i[1] == hy:
                adleft = 1 # adjoining left square has snake body
            if i[0] - hx == 1 and i[1] == hy:
                adright = 1 # adjoining right square has snake body
        
        return (fdx, fdy, awx, awy, adtop, adbot, adleft, adright)