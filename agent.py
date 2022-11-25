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
        for action in action_list:
            if self.N[s_prime][action] < self.Ne:
                argmax_list[action] = 1
            else:
                argmax_list[action] = self.Q[s_prime][action]
        
        # Reverse list to match action_list indices
        argmax_list.reverse()
        
        # Choose the action at s_prime that maximizes it's Q-value
        max = argmax_list[0]
        best_action = action_list[0]
        for idx, val in enumerate(argmax_list):
            if val > max:
                max = val
                best_action = action_list[idx]

        return best_action

        # Helper function to determine if snake dies
    def snake_dead(self, s_prime, a_prime):
        # state = (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        
        # If snake runs into a wall
        if (s_prime[2] == 1 and a_prime == utils.LEFT) or (s_prime[2] == 2 and a_prime == utils.RIGHT) or (s_prime[3] == 1 and a_prime == utils.UP) or (s_prime[3] == 2 and a_prime == utils.DOWN):
            return True
        
        # If snake runs into it's own body
        if (s_prime[4] == 1 and a_prime == utils.UP) or (s_prime[5] == 1 and a_prime == utils.DOWN) or (s_prime[6] == 1 and a_prime == utils.LEFT) or (s_prime[7] == 1 and a_prime == utils.RIGHT):
            return True

        return False

    def calculate_reward(self, s_prime, points, dead):
        # 2. From the result of the action on the environment, the agent obtains a reward r_t
        r_t = -.1
        if dead:
            r_t = -1
        elif points == self.points + 1: # if food pellet found
            r_t = 1
        return r_t

    def discretize_new_environment(self, environment, a_prime):
        # 3. The agent then “discretizes” this new environment by generating a state based off of the new, post-action environment
        if a_prime == utils.RIGHT:
            environment[0] += 1
        elif a_prime == utils.LEFT:
            environment[0] -= 1
        elif a_prime == utils.DOWN:
            environment[1] += 1
        elif a_prime == utils.UP:
            environment[1] -= 1
        
        return self.generate_state(environment)

    def update_q_n(self, s_prime, a_prime, dead):
        # 4. With s_t, a_t, r_t, and s_t+1, the agent can update its Q-value estimate for the state action pair Q(s_t, a_t)
        # a. Update N(s_t, a_t)
        self.N[s_prime][a_prime] += 1

        # b. Update Q(s_t, a_t)
        alpha = self.C / ( self.C + self.N[self.s][self.a] )     
        if dead:
            self.Q[self.s][self.a] = self.Q[self.s][self.a] + alpha * (self.points - self.Q[self.s][self.a] + self.gamma * 0 )
        else:
            self.Q[self.s][self.a] = self.Q[self.s][self.a] + alpha * (self.points - self.Q[self.s][self.a] + self.gamma * self.Q[s_prime][a_prime] )


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
        # state = (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        s_prime = self.generate_state(environment)

        # TODO Check that the snake is not past the border walls
        # TODO Use points input to figure out reward -> update points if food found
        
        if dead and self._train: # if snake dies, don't look for optimal next step
            self.update_q_n(s_prime, None, dead)
            self.reset()
            return self.a

        # 1. Choose optimal action based on Q-value (or lack of exploration)
        a_prime = self.find_best_action(s_prime)
        # 2. From the result of the action on the environment, the agent obtains a reward r_t
        r_t = self.calculate_reward(s_prime, points, dead)
        
        # When t = 0, initialize state and action, disregarding N and Q-table updating
        if self.s == None and self.a == None:
            self.s = s_prime
            self.a = a_prime
            return a_prime
            # Do we update points in this situation?

        # 3. TODO The agent then “discretizes” this new environment by generating a state based off of the new, post-action environment
        # s_double_prime = self.discretize_new_environment(environment, a_prime)
        # r_t2 = self.calculate_reward(s_double_prime, dead, points)

        # 4. With s_t, a_t, r_t, and s_t+1, the agent can update its Q-value estimate for the state action pair Q(s_t, a_t)
        if self._train: self.update_q_n(s_prime, a_prime, dead)
        # 5. The agent is now in state s_t+1, and the process repeats
        if r_t == 1: self.points += r_t # if food pellet found, increment global points
        self.s = s_prime
        self.a = a_prime

        return a_prime


    # Helper function to generate a state given an environment 
    # Each state in the MDP is a tuple of the form returned
    def generate_state(self, environment):

        snake_head_x = environment[0]
        snake_head_y = environment[1]
        snake_body = environment[2]
        food_x = environment[3]
        food_y = environment[4]
        
        food_dir_x = 0
        if snake_head_x > food_x:
            food_dir_x = 1 # if food on left of snake head
        elif snake_head_x < food_x:
            food_dir_x = 2 # if food on right of snake head
        
        food_dir_y = 0
        if snake_head_y > food_y:
            food_dir_y = 1 # if food above snake head
        elif snake_head_y < food_y:
            food_dir_y = 2 # if food below snake head

        adjoining_wall_x = 0
        if snake_head_x == 1:
            adjoining_wall_x = 1 # if wall on left of head
        if snake_head_x == utils.DISPLAY_WIDTH - 2:
            adjoining_wall_x = 2 # if wall on right of head

        adjoining_wall_y = 0
        if snake_head_y == 1:
            adjoining_wall_y = 1 # if wall above head
        if snake_head_y == utils.DISPLAY_HEIGHT - 2:
            adjoining_wall_x = 2 # if wall below head

        # Check where the snake's body is in relation to it's head
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        for coord in snake_body:
            if coord[1] - snake_head_y == 1 and coord[0] == snake_head_x:
                adjoining_body_top = 1
            if coord[1] - snake_head_y == -1 and coord[0] == snake_head_x:
                adjoining_body_bottom = 1
            if coord[0] - snake_head_x == 1 and coord[1] == snake_head_y:
                adjoining_body_left = 1
            if coord[0] - snake_head_x == -1 and coord[1] == snake_head_y:
                adjoining_body_right = 1

        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)