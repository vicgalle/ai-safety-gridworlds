import copy
import pickle
import numpy as np

from absl import app
from absl import flags

from ai_safety_gridworlds.environments.shared import safety_game
from ai_safety_gridworlds.environments.shared import safety_ui
from ai_safety_gridworlds.environments.friend_foe import FriendFoeEnvSimple


bandit_type = 'adversary'
environment_data = {}
extra_step = False

env = FriendFoeEnvSimple(environment_data=environment_data,
                             bandit_type=bandit_type,
                             extra_step=extra_step)

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