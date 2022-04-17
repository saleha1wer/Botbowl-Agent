from hashlib import new
from types import NoneType
import numpy as np

from convert_action import short_to_normal



# class short_action():
#     def __init__(self,type,) -> None:
#         pass
        

mask = np.load('mask.npy', allow_pickle=True)
print(len(mask))
new_mask = []
old_type = None
count = 0
print(len(mask))
for pair in mask:
    print(pair[1])
    if pair[1] is not None:
        action = pair[1][0]
        if action.action_type != old_type:
            # print('action',action.action_type,count)
            count = 0
            new_mask.append(action.action_type)
            old_type = action.action_type
        else:
            count = count + 1
    else:
        action = None
        new_mask.append(action)
print(new_mask)
np.save('short_mask',new_mask,allow_pickle=True)
print(new_mask)
""""    
0 - 19: non-spacial actions, (no end setup)
20: None
21: None
22: PLACE_BALL,
23: PUSH,
24: FOLLOW_UP,
25: MOVE,
26: BLOCK,
27: PASS,
28: FOUL,
29: HANDOFF,
30: LEAP,
31: STAB,
32: SELECT_PLAYER,
33: START_MOVE,
34: START_BLOCK,
35: START_BLITZ,
36: START_PASS,
37: START_FOUL,
38: START_HANDOFF

"""