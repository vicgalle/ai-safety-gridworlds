import copy
import pickle
import numpy as np
from numpy.random import choice


from absl import app
from absl import flags

from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared import safety_ui
from ai_safety_gridworlds.environments.friend_foe import FriendFoeEnvSimple

class Agent():
    """
    Parent abstract Agent.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        """
        This implements the policy, \pi : S -> A.
        obs is the observed state s
        """
        raise NotImplementedError()

    def update(self, obs, actions, rewards, new_obs):
        """
        This is after an interaction has ocurred, ie all agents have done their respective actions, observed their rewards and arrived at
        a new observation (state).
        For example, this is were a Q-learning agent would update her Q-function
        """
        pass

class IndQLearningAgent(Agent):
    """
    A Q-learning agent that treats other players as part of the environment (independent Q-learning).
    She represents Q-values in a tabular fashion, i.e., using a matrix Q.
    Intended to use as a baseline
    """

    def __init__(self, action_space, n_states, learning_rate, epsilon, gamma):
        Agent.__init__(self, action_space)

        self.n_states = n_states
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        # This is the Q-function Q(s, a)
        self.Q = np.zeros([self.n_states, len(self.action_space)])

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def act(self, obs=None):
        """An epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return choice(self.action_space)
        else:
            #print('---')
            #print(obs)
            #print(self.Q)
            #print(np.argmax(self.Q[obs, :]))
            if obs == None:
                obs = 12
            #return choice(self.action_space, p=self.softmax(self.Q[obs,:]))
            return self.action_space[np.argmax(self.Q[obs, :])]

    def update(self, obs, actions, rewards, new_obs):
        """The vanilla Q-learning update rule"""
        a0, _ = actions
        r0, _ = rewards

        self.Q[obs, a0] = (1 - self.alpha)*self.Q[obs, a0] + self.alpha*(r0 + self.gamma*np.max(self.Q[new_obs, :]))


bandit_type = 'neutral'
environment_data = {}
extra_step = False

env = FriendFoeEnvSimple(environment_data=environment_data,
                             bandit_type=bandit_type,
                             extra_step=extra_step)

test = False
if test:
    o = env.step([1])  #First action doesnt matter
    print(o)
    o = env.step([0])
    print(o)
    o = env.step([0])
    print(o)
    o = env.step([0])
    print(o)
    o = env.step([2])  # Left
    # print('.......')
    #print('Ov. perf', env.get_overall_performance())
    print(o)

    print('----------')


    o = env.step([1])  #First action doesnt matter
    print(o)
    o = env.step([0])
    print(o)
    o = env.step([0])
    print(o)
    o = env.step([0])
    print(o)
    o = env.step([2])  # Left
    # print('.......')
    #print('Ov. perf', env.get_overall_performance())
    print(o)



batch_size = 1
gamma = 0.95

# Reward matrix for the Iterated Prisoner's Dilemma
#ipd_rewards = np.array([[-1., 0.], [-3., -2.]])


#env = RMG(max_steps=max_steps, payouts=ipd_rewards, batch_size=batch_size)
env.reset()

possible_actions = [0, 1, 2, 3]  # 4 dirs

#cooperator, defector = RandomAgent(possible_actions, p=0.9), RandomAgent(possible_actions, p=0.1)
p0 = IndQLearningAgent(possible_actions, n_states=13, learning_rate=0.05, epsilon=0.99, gamma=gamma)


#cooperator, defector = IndQLearningAgent(possible_actions, n_states=1, learning_rate=0.05, epsilon=0.1, gamma=gamma), \
#    IndQLearningAgent(possible_actions, n_states=1, learning_rate=0.05, epsilon=0.1, gamma=gamma)


# Stateless interactions (agents do not have memory)
s = 0

n_iter = 40000

r0s = []
r1s = []

for i in range(n_iter):

    # A full episode:
    done = False

    while not done:

        # Agents decide
        a0 = p0.act()
        #print(a0)
        #a1 = p1.act()

        # World changes
        #_, (r0, r1), done, _ = env.step(([a0], [a1]))
        new_s, (r0, _), done, info = env.step([a0])
        #print('S:', new_s)
        #print('r:', r0)
        #print('I:', i)
        # Agents learn

        p0.update(s, (a0, _), (r0, _), new_s )
        #p1.update(s, (a1, a0), (r1, r0), s )

        s = new_s  

        #print(r0, r1)
        r0s.append(r0)
        #r1s.append(r1[0])

    if i % 10 == 1:
        p0.epsilon *= 0.999
    print('eps', p0.epsilon)
    print('Q', p0.Q)
    env.reset()
