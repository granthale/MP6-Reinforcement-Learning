import numpy as np
import utils

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
        argmax_list = [0,0,0,0]
        action_list = [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]
        if self._train: # WITH EXPLORATION
            for action in action_list:
                if self.N[s_prime][action] < self.Ne: # if state and action haven't been explored sufficiently during training
                    argmax_list[action] = 1
                else:
                    argmax_list[action] = self.Q[s_prime][action]

        else: # WITHOUT EXPLORATION
            for action in action_list:
                argmax_list[action] = self.Q[s_prime][action]

        # Reverse list to match action_list indices : in case of ties, this prioritizes R -> L -> D -> U
        argmax_list.reverse()
        
        max = argmax_list[0]
        best_action = action_list[0]
        for idx, val in enumerate(argmax_list):
            if val > max:
                max = val
                best_action = action_list[idx]

        return best_action

    def calculate_reward(self, points, dead):
        # 2. From the result of the action on the environment, the agent obtains a reward r_t
        r_t = -.1
        if dead:
            r_t = -1
        elif points > self.points: # if food pellet found
            r_t = 1
        return r_t

    def update_q_n(self, r_t, s_prime, a_prime):
        # a. Update N(s_t, a_t)
        self.N[self.s][self.a] += 1

        # b. Update Q(s_t, a_t)
        alpha = self.C / ( self.C + self.N[self.s][self.a] )
        if r_t == -1: # if dead, assume no next move
            self.Q[self.s][self.a] += alpha * (r_t - self.Q[self.s][self.a])
        else:
            self.Q[self.s][self.a] += alpha * (r_t + self.gamma * self.Q[s_prime][a_prime] - self.Q[self.s][self.a] )


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
        # 2. From the result of the previous action on the environment (self.s, self.a), the agent obtains a reward r_t
        r_t = self.calculate_reward(points, dead)
        if dead and self._train: # if snake dies, don't look for the optimal next step
            self.update_q_n(r_t, None, None)
            self.reset()
            return None

        # 3. The agent “discretizes” this new environment by generating a state based off of the new, post-action environment
        s_prime = self.generate_state(environment)

        # TODO Check that the snake is not past the border walls?
        
        # 1. Choose optimal action based on Q-value (or lack of exploration)
        a_prime = self.find_best_action(s_prime)
        # When t = 0, initialize state and action, disregarding N and Q-table updating
        if self.s == None and self.a == None:
            self.s = s_prime
            self.a = a_prime
            return a_prime

        # 4. With s_t, a_t, r_t, and s_t+1, the agent can update its Q-value estimate for the state action pair Q(s_t, a_t)
        if self._train: self.update_q_n(r_t, s_prime, a_prime)
        
        # 5. The agent is now in state s_t+1, and the process repeats
        if r_t == 1: # if food pellet found
            self.points += r_t # increment global points
        self.s = s_prime
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