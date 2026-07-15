"""
CartPole with continuous action space: classic balancing task 
-------------------
Description:
A pole is hinged on a cart that slides along a rail. The agent pushed the cart
left or right and must keep the pole upright 

The action: continuous force [-1,1] scaled to += 10 N 
Observation: [cart position, cart velocity, pole angle, pole angular velocity]
Reward: +1 for every step the pole remains upright
Terminated: The pole falls over (|angle| > 12 degrees) or the cart goes out of bounds (position > 2.4)
truncated: The episode length exceeds 500 steps
"""


import numpy as np 
