# -*- coding: utf-8 -*-
"""
Created on Mon Jul 3 18:59:17 2022

@author: divas
"""

import pickle
from smoothness import *

with open("saved_actions_r1.pk", 'rb') as handle:
    chosen_actions = pickle.load(handle)
    
    
ans_j = dimensionless_jerk(chosen_actions, 30, 10)
ans_sal = spectral_arclength(chosen_actions, 30)

with open("answer.pk", "wb") as handle:
    pickle.dump(ans_j, handle)
    pickle.dump(ans_sal, handle)