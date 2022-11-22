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
    
    # Helper function to determine if food pellet has been found
    def food_pellet_found(self, environment, a_prime):
        if a_prime == utils.RIGHT:
            head_x = environment[0] + 1
        elif a_prime == utils.LEFT:
            head_x = environment[0] - 1
        elif a_prime == utils.DOWN:
            head_y = environment[1] + 1
        elif a_prime == utils.UP:
            head_y = environment[1] - 1
        
        if (head_x, head_y) == (environment[3], environment[4]):
            return True
        return False

    # TODO Helper function to determine if snake dies
    def snake_dead(self, environment, a_prime):
        
        return False
    
    # Helper function to retrieve n
    def get_n(self, state, action):
        return self.N[state[0]][state[1]][state[2]][state[3]][state[4]][state[5]][state[6]][state[7]][action]

    # Helper function to retrieve q
    def get_q(self, state, action):
        return self.Q[state[0]][state[1]][state[2]][state[3]][state[4]][state[5]][state[6]][state[7]][action]

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

        # TODO t = 0

        # 1. Choose optimal action based on Q-value or lack of exploration
            # action = argmax( f(Q(s,a), N(s,a)) )
                # f( Q(s,a), N(s,a) ) = 1 if N(s,a) < N_e
                # else                = Q(s,a)
        argmax_list = [0,0,0,0]
        action_list = [utils.RIGHT, utils.LEFT, utils.DOWN, utils.UP]
        for action in action_list:
            if self.get_n(s_prime, action) < self.Ne:
                argmax_list[action] = 1
            else:
                argmax_list[action] = self.get_q(s_prime, action)

        # Chooses the action at s_prime that maximizes it's Q-value (or explores)
        a_prime = argmax_list.index(max(argmax_list))

        # 2. From the result of the action on the environment, the agent obtains a reward r_t
            # if food_pellet_found: 
                # r_t = 1
            # if snake_dies: 
                # r_t = -1
                # and self.reset()
            # else: 
                # r_t = -0.1
            # Update points
        
        # TODO self.a or a_prime?
        r_t = -0.1
        if self.food_pellet_found(environment, self.a):
            r_t = 1
        elif self.snake_dead(environment, self.a): # TODO
            r_t = -1
            self.reset()

        # 3. The agent then “discretizes” this new environment by generating a state based off of the new, post-action environment
        if a_prime == utils.RIGHT:
            environment[0] += 1
        elif a_prime == utils.LEFT:
            environment[0] -= 1
        elif a_prime == utils.DOWN:
            environment[1] += 1
        elif a_prime == utils.UP:
            environment[1] -= 1
        s_double_prime = self.generate_state(environment)

        # 4. With s_t, a_t, r_t, and s_t+1, the agent can update its Q-value estimate for the state action pair Q(s_t, a_t)
            # a. Update N(s_t, a_t)
            # b. Update Q(s_t, a_t)

        self.N[s_prime[0]][s_prime[1]][s_prime[2]][s_prime[3]][s_prime[4]][s_prime[5]][s_prime[6]][s_prime[7]][a_prime] += 1
        
        # TODO s_prime or self.s?
        alpha = self.C / (self.C + self.get_n(s_prime, a_prime))
        self.Q[self.s[0]][self.s[1]][self.s[2]][self.s[3]][self.s[4]][self.s[5]][self.s[6]][self.s[7]][self.a] = self.get_q(self.s, self.a) + alpha * (r_t - self.get_q(self.s, self.a) + self.gamma * self.get_q(s_prime, a_prime) )

        # 5. TODO The agent is now in state s_t+1, and the process repeats
        self.s = s_prime
        self.a = a_prime

        return utils.RIGHT

    # Helper function to generate a state given an environment 
    # Each state in the MDP is a tuple of the form returned
    def generate_state(self, environment):

        snake_head_x = environment[0]
        snake_head_y = environment[1]
        snake_body = environment[2]
        food_x = environment[3]
        food_y = environment[4]
        
        food_dir_x = 0
        if snake_head_x < food_x:
            food_dir_x = 1 # if head on the left
        elif snake_head_x > food_x:
            food_dir_x = 2 # if head on the right
        
        food_dir_y = 0
        if snake_head_y < food_y:
            food_dir_y = 1 # if head below food
        elif snake_head_y > food_y:
            food_dir_y = 2 # if head above food

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