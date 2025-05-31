from openai import OpenAI
import torch
from utils.obs2text import gen_text_desc

client = OpenAI()

ID_TO_ACTION = {0: 'left turn', 1: 'right turn', 2: 'move forward', 3: 'pick up', 4: 'drop', 5: 'toggle'}

prefix = {}

prefix['MiniGrid-Dynamic-Obstacles-6x6-v0'] = '''Imagine now you are a reinforcement learning agent in a 2D gridworld RL platform called MiniGrid, and you are learning to complete tasks in a specific RL environment called 'Dynamic Obstacles' on this Minigrid platform. This 'Dynamic Obstacles' environment is an empty room with moving obstacles (i.e. blue balls). The goal of the agent is to reach the green goal square using as few time steps as possible without colliding with any obstacle (i.e. blue ball). A large penalty is subtracted if the agent collides with an obstacle and the episode finishes. Your possible actions as an agent at each time step are: 'turn left', 'turn right', 'move forward'.
    
You are provided with 5 successful trajectories of expert demonstrations of the oracle courses of actions to complete tasks in this 'Dynamic Obstacles' environment for your reference, which are listed in detail below:

Expert Demonstration Trajectory 1:

Goal of the agent: get to the green goal square

Observation 1: You see a wall 4 steps forward, You see a wall 1 step left, You see a blue ball 1 step right, You see a blue ball 2 steps right and 3 steps forward, You see a green goal 3 steps right and 3 steps forward, You see a blue ball 3 steps right.
Action 1: move forward.
Observation 2: You see a wall 1 step left, You see a blue ball 1 step forward, You see a blue ball 2 steps right, You see a green goal 3 steps right and 3 steps forward, You see a blue ball 3 steps right and 2 steps forward.
Action 2: turn right.
Observation 3: You see a wall 4 steps forward, You see a wall 1 step right, You see a green goal 3 steps left and 3 steps forward, You see a blue ball 1 step left and 3 steps forward, You see a blue ball 1 step left and 2 steps forward, You see a blue ball 1 step left and 1 step forward.
Action 3: move forward.
Observation 4: You see a wall 3 steps forward, You see a wall 1 step right, You see a green goal 3 steps left and 2 steps forward, You see a blue ball 2 steps left and 2 steps forward, You see a blue ball 2 steps left and 1 step forward, You see a blue ball 2 steps left.
Action 4: move forward.
Observation 5: You see a wall 2 steps forward, You see a wall 1 step right, You see a green goal 3 steps left and 1 step forward, You see a blue ball 3 steps left, You see a blue ball 1 step left.
Action 5: move forward.
Observation 6: You see a wall 1 step right, You see a green goal 3 steps left and 1 step forward, You see a blue ball 1 step forward.
Action 6: turn left.
Observation 7: You see a wall 4 steps forward, You see a wall 3 steps left, You see a wall 2 steps right, You see a blue ball 2 steps left and 3 steps forward, You see a blue ball 1 step left and 1 step forward, You see a green goal 1 step right and 3 steps forward, You see a blue ball 1 step right and 1 step forward.
Action 7: move forward.
Observation 8: You see a wall 3 steps left, You see a wall 2 steps right, You see a blue ball 2 steps left and 2 steps forward, You see a blue ball 2 steps forward, You see a blue ball 1 step forward, You see a green goal 1 step right and 3 steps forward.
Action 8: turn right.
Observation 9: You see a wall 2 steps forward, You see a wall 1 step right, You see a green goal 3 steps left and 1 step forward, You see a blue ball 3 steps left.
Action 9: move forward.
Observation 10: You see a wall 1 step forward, You see a wall 1 step right, You see a green goal 3 steps left.
Action 10: turn left.
Observation 11: You see a wall 1 step right, You see a blue ball 3 steps left and 1 step forward, You see a blue ball 2 steps left, You see a blue ball 1 step left and 2 steps forward, You see a green goal 3 steps forward.
Action 11: move forward.
Observation 12: You see a wall 1 step right, You see a blue ball 3 steps left, You see a blue ball 2 steps left, You see a blue ball 1 step left, You see a green goal 2 steps forward.
Action 12: move forward.
Observation 13: You see a wall 1 step right, You see a blue ball 2 steps left, You see a blue ball 1 step left and 1 step forward, You see a green goal 2 steps forward, You see a blue ball 1 step forward.
Action 13: turn left.
Observation 14: You see a wall 4 steps forward, You see a wall 2 steps left, You see a blue ball 1 step left and 2 steps forward, You see a blue ball 1 step right and 2 steps forward, You see a blue ball 2 steps right and 1 step forward, You see a green goal 2 steps right.
Action 14: turn right.
Observation 15: You see a wall 1 step right, You see a blue ball 2 steps left and 2 steps forward, You see a blue ball 2 steps left, You see a green goal 2 steps forward, You see a blue ball 1 step forward.
Action 15: turn left.
Observation 16: You see a wall 2 steps left, You see a blue ball 1 step forward, You see a blue ball 1 step right and 2 steps forward, You see a blue ball 1 step right and 1 step forward, You see a green goal 2 steps right.
Action 16: turn right.
Observation 17: You see a wall 1 step right, You see a blue ball 3 steps left and 1 step forward, You see a blue ball 1 step left and 2 steps forward, You see a blue ball 1 step left and 1 step forward, You see a green goal 2 steps forward.
Action 17: move forward.
Observation 18: You see a wall 1 step right, You see a blue ball 3 steps left, You see a blue ball 2 steps left and 1 step forward, You see a green goal 2 steps forward, You see a blue ball 1 step forward.
Action 18: turn left.
Observation 19: You see a wall 2 steps left, You see a blue ball 3 steps forward, You see a blue ball 1 step right and 3 steps forward, You see a blue ball 1 step right and 1 step forward, You see a green goal 2 steps right.
Action 19: turn right.
Observation 20: You see a wall 1 step right, You see a blue ball 3 steps left and 2 steps forward, You see a blue ball 1 step left and 2 steps forward, You see a green goal 2 steps forward.
Action 20: move forward.
Observation 21: You see a wall 1 step right, You see a blue ball 2 steps left and 1 step forward, You see a blue ball 1 step left, You see a green goal 1 step forward.
Action 21: move forward.
Observation 22: You have successfully reached the goal.

Expert Demonstration Trajectory 2:

Goal of the agent: get to the green goal square

Observation 1: You see a wall 1 step left, You see a blue ball 1 step forward, You see a blue ball 2 steps right and 3 steps forward, You see a green goal 3 steps right and 3 steps forward, You see a blue ball 3 steps right and 2 steps forward.
Action 1: turn right.
Observation 2: You see a wall 4 steps forward, You see a wall 1 step right, You see a green goal 3 steps left and 3 steps forward, You see a blue ball 2 steps left and 2 steps forward, You see a blue ball 2 steps left, You see a blue ball 1 step left and 2 steps forward.
Action 2: move forward.
Observation 3: You see a wall 3 steps forward, You see a wall 1 step right, You see a green goal 3 steps left and 2 steps forward, You see a blue ball 3 steps left, You see a blue ball 2 steps left, You see a blue ball 1 step left.
Action 3: move forward.
Observation 4: You see a wall 2 steps forward, You see a wall 1 step right, You see a green goal 3 steps left and 1 step forward.
Action 4: move forward.
Observation 5: You see a wall 1 step forward, You see a wall 1 step right, You see a green goal 3 steps left.
Action 5: turn left.
Observation 6: You see a wall 1 step right, You see a blue ball 3 steps left and 1 step forward, You see a blue ball 1 step left and 2 steps forward, You see a blue ball 1 step left and 1 step forward, You see a green goal 3 steps forward.
Action 6: move forward.
Observation 7: You see a wall 1 step right, You see a blue ball 2 steps left and 2 steps forward, You see a blue ball 2 steps left, You see a green goal 3 steps forward, You see a blue ball 1 step forward.
Action 7: turn left.
Observation 8: You see a wall 1 step left, You see a blue ball 3 steps forward, You see a blue ball 1 step right and 1 step forward, You see a blue ball 3 steps right and 3 steps forward, You see a green goal 3 steps right.
Action 8: turn right.
Observation 9: You see a wall 1 step right, You see a blue ball 3 steps left and 2 steps forward, You see a blue ball 2 steps left and 1 step forward, You see a blue ball 2 steps left, You see a green goal 3 steps forward.
Action 9: move forward.
Observation 10: You see a wall 1 step right, You see a blue ball 2 steps left and 1 step forward, You see a green goal 2 steps forward.
Action 10: move forward.
Observation 11: You see a wall 1 step right, You see a blue ball 1 step left, You see a green goal 1 step forward.
Action 11: move forward.
Observation 12: You have successfully reached the goal.

Expert Demonstration Trajectory 3:

Goal of the agent: get to the green goal square

Observation 1: You see a wall 4 steps forward, You see a wall 1 step left, You see a blue ball 1 step right and 2 steps forward, You see a blue ball 2 steps right and 2 steps forward, You see a blue ball 2 steps right, You see a green goal 3 steps right and 3 steps forward.
Action 1: move forward.
Observation 2: You see a wall 3 steps forward, You see a wall 1 step left, You see a blue ball 1 step right, You see a green goal 3 steps right and 2 steps forward, You see a blue ball 3 steps right and 1 step forward.
Action 2: move forward.
Observation 3: You see a wall 1 step left, You see a blue ball 1 step forward, You see a blue ball 2 steps right, You see a green goal 3 steps right and 2 steps forward, You see a blue ball 3 steps right.
Action 3: turn right.
Observation 4: You see a wall 4 steps forward, You see a wall 3 steps left, You see a wall 2 steps right, You see a green goal 2 steps left and 3 steps forward, You see a blue ball 2 steps left and 1 step forward, You see a blue ball 1 step left and 3 steps forward, You see a blue ball 1 step left and 2 steps forward.
Action 4: move forward.
Observation 5: You see a wall 3 steps left, You see a wall 2 steps right, You see a green goal 2 steps left and 2 steps forward, You see a blue ball 2 steps forward, You see a blue ball 1 step forward.
Action 5: turn left.
Observation 6: You see a wall 2 steps left, You see a wall 3 steps right, You see a blue ball 1 step forward, You see a green goal 2 steps right and 2 steps forward.
Action 6: turn right.
Observation 7: You see a wall 3 steps left, You see a wall 2 steps right, You see a green goal 2 steps left and 2 steps forward, You see a blue ball 2 steps left and 1 step forward, You see a blue ball 2 steps forward, You see a blue ball 1 step forward.
Action 7: turn left.
Observation 8: You see a wall 2 steps left, You see a wall 3 steps right, You see a blue ball 2 steps forward, You see a blue ball 1 step forward, You see a blue ball 1 step right and 1 step forward, You see a green goal 2 steps right and 2 steps forward.
Action 8: turn right.
Observation 9: You see a wall 3 steps forward, You see a wall 3 steps left, You see a wall 2 steps right, You see a green goal 2 steps left and 2 steps forward, You see a blue ball 2 steps left and 1 step forward, You see a blue ball 1 step left and 1 step forward.
Action 9: move forward.
Observation 10: You see a wall 2 steps forward, You see a wall 3 steps left, You see a wall 2 steps right, You see a green goal 2 steps left and 1 step forward, You see a blue ball 1 step left and 1 step forward.
Action 10: move forward.
Observation 11: You see a wall 1 step forward, You see a wall 2 steps right, You see a green goal 2 steps left.
Action 11: turn left.
Observation 12: You see a wall 1 step right, You see a blue ball 2 steps left, You see a blue ball 1 step left, You see a green goal 2 steps forward, You see a blue ball 1 step forward.
Action 12: turn left.
Observation 13: You see a wall 4 steps forward, You see a wall 2 steps left, You see a blue ball 1 step left and 2 steps forward, You see a blue ball 1 step left and 1 step forward, You see a blue ball 1 step right and 1 step forward, You see a green goal 2 steps right.
Action 13: turn right.
Observation 14: You see a wall 1 step right, You see a blue ball 2 steps left, You see a green goal 2 steps forward.
Action 14: move forward.
Observation 15: You see a wall 1 step right, You see a green goal 1 step forward.
Action 15: move forward.
Observation 16: You have successfully reached the goal.

Expert Demonstration Trajectory 4:

Goal of the agent: get to the green goal square

Observation 1: You see a wall 4 steps forward, You see a wall 1 step left, You see a blue ball 1 step right, You see a blue ball 2 steps right and 3 steps forward, You see a green goal 3 steps right and 3 steps forward, You see a blue ball 3 steps right and 2 steps forward.
Action 1: move forward.
Observation 2: You see a wall 1 step left, You see a blue ball 1 step forward, You see a blue ball 1 step right and 3 steps forward, You see a blue ball 2 steps right and 1 step forward, You see a green goal 3 steps right and 3 steps forward.
Action 2: turn right.
Observation 3: You see a wall 1 step right, You see a green goal 3 steps left and 3 steps forward, You see a blue ball 3 steps left and 2 steps forward, You see a blue ball 3 steps forward, You see a blue ball 1 step forward.
Action 3: turn left.
Observation 4: You see a wall 4 steps forward, You see a wall 1 step left, You see a blue ball 2 steps right, You see a green goal 3 steps right and 3 steps forward, You see a blue ball 3 steps right and 2 steps forward, You see a blue ball 3 steps right and 1 step forward.
Action 4: move forward.
Observation 5: You see a wall 3 steps forward, You see a wall 1 step left, You see a blue ball 2 steps right and 2 steps forward, You see a green goal 3 steps right and 2 steps forward, You see a blue ball 3 steps right and 1 step forward.
Action 5: move forward.
Observation 6: You see a wall 2 steps forward, You see a wall 1 step left, You see a blue ball 2 steps right, You see a green goal 3 steps right and 1 step forward.
Action 6: move forward.
Observation 7: You see a wall 1 step forward, You see a wall 1 step left, You see a green goal 3 steps right.
Action 7: turn right.
Observation 8: You see a wall 1 step left, You see a green goal 3 steps forward, You see a blue ball 1 step right and 3 steps forward, You see a blue ball 2 steps right and 3 steps forward, You see a blue ball 3 steps right and 2 steps forward.
Action 8: move forward.
Observation 9: You see a wall 1 step left, You see a green goal 2 steps forward, You see a blue ball 1 step right and 1 step forward, You see a blue ball 2 steps right and 1 step forward, You see a blue ball 2 steps right.
Action 9: move forward.
Observation 10: You see a wall 1 step left, You see a green goal 1 step forward, You see a blue ball 2 steps right and 1 step forward, You see a blue ball 2 steps right, You see a blue ball 3 steps right.
Action 10: move forward.
Observation 11: You have successfully reached the goal.

Expert Demonstration Trajectory 5:

Goal of the agent: get to the green goal square

Observation 1: You see a wall 1 step left, You see a blue ball 2 steps forward, You see a blue ball 2 steps right and 3 steps forward, You see a blue ball 2 steps right, You see a green goal 3 steps right and 3 steps forward.
Action 1: move forward.
Observation 2: You see a wall 1 step left, You see a blue ball 2 steps forward, You see a blue ball 1 step right and 1 step forward, You see a blue ball 2 steps right, You see a green goal 3 steps right and 2 steps forward.
Action 2: move forward.
Observation 3: You see a wall 2 steps forward, You see a wall 1 step left, You see a blue ball 1 step right and 1 step forward, You see a green goal 3 steps right and 1 step forward, You see a blue ball 3 steps right.
Action 3: move forward.
Observation 4: You see a wall 1 step left, You see a blue ball 1 step forward, You see a blue ball 2 steps right, You see a green goal 3 steps right and 1 step forward.
Action 4: turn right.
Observation 5: You see a wall 2 steps left, You see a wall 3 steps right, You see a green goal 1 step left and 3 steps forward, You see a blue ball 1 step left and 1 step forward, You see a blue ball 1 step forward, You see a blue ball 1 step right and 2 steps forward.
Action 5: turn left.
Observation 6: You see a wall 2 steps forward, You see a wall 1 step left, You see a blue ball 1 step right, You see a green goal 3 steps right and 1 step forward.
Action 6: move forward.
Observation 7: You see a wall 1 step forward, You see a wall 1 step left, You see a blue ball 1 step right, You see a green goal 3 steps right.
Action 7: turn right.
Observation 8: You see a wall 1 step left, You see a green goal 3 steps forward, You see a blue ball 1 step right and 1 step forward, You see a blue ball 1 step right, You see a blue ball 3 steps right and 2 steps forward.
Action 8: move forward.
Observation 9: You see a wall 1 step left, You see a green goal 2 steps forward, You see a blue ball 2 steps right, You see a blue ball 3 steps right and 2 steps forward.
Action 9: move forward.
Observation 10: You see a wall 1 step left, You see a green goal 1 step forward, You see a blue ball 2 steps right and 1 step forward.
Action 10: move forward.
Observation 11: You have successfully reached the goal.

Currently, as the reinforcement learning agent, you are following the following list of strategies when making action decisions in this 'Dynamic Obstacles' environment:\n\n'''

prefix['BabyAI-PutNextS6N3-v0'] = '''Imagine now you are a reinforcement learning agent in a 2D gridworld RL platform called BabyAI, and you are learning to complete tasks in a specific RL environment called 'Put Next' on this BabyAI platform. In each run of this 'Put Next' task in this RL environment, your goal as an agent is to move a designated object next to another designated object using as few time steps as possible. Your possible actions as an agent at each time step are: 'turn left', 'turn right', 'move forward', 'pick up', 'drop'.
    
You are provided with 5 successful trajectories of expert demonstrations of the oracle courses of actions to complete tasks in this 'Put Next' environment for your reference, which are listed in detail below:

Expert Demonstration Trajectory 1:

Goal of the agent: put the grey key next to the yellow ball

Observation 1: You see a wall 2 steps forward, You see a wall 3 steps left, You see a wall 2 steps right.
Action 1: left turn.
Observation 2: You see a wall 3 steps forward, You see a wall 2 steps right, You see a grey key 2 steps left and 2 steps forward, You see a purple ball 1 step left and 2 steps forward, You see a grey box 1 step left and 1 step forward.
Action 2: left turn.
Observation 3: You see a wall 2 steps left, You see a wall 3 steps right, You see a blue ball 1 step left and 5 steps forward, You see a red box 1 step left and 4 steps forward, You see a yellow ball 1 step right and 5 steps forward, You see a grey box 1 step right and 1 step forward, You see a grey key 2 steps right and 2 steps forward, You see a purple ball 2 steps right and 1 step forward.
Action 3: move forward.
Observation 4: You see a wall 2 steps left, You see a blue ball 1 step left and 4 steps forward, You see a red box 1 step left and 3 steps forward, You see a yellow ball 1 step right and 4 steps forward, You see a grey box 1 step right, You see a grey key 2 steps right and 1 step forward, You see a purple ball 2 steps right.
Action 4: move forward.
Observation 5: You see a wall 6 steps forward, You see a wall 2 steps left, You see a blue ball 1 step left and 3 steps forward, You see a red box 1 step left and 2 steps forward, You see a yellow ball 1 step right and 3 steps forward, You see a grey key 2 steps right.
Action 5: right turn.
Observation 6: You see a yellow ball 3 steps left and 1 step forward, You see a grey key 2 steps forward, You see a purple ball 1 step right and 2 steps forward, You see a grey box 1 step right and 1 step forward.
Action 6: move forward.
Observation 7: You see a yellow ball 3 steps left, You see a grey key 1 step forward, You see a purple ball 1 step right and 1 step forward, You see a grey box 1 step right.
Action 7: pick up.
Observation 8: You carry a grey key, You see a wall 2 steps forward, You see a yellow ball 3 steps left, You see a purple ball 1 step right and 1 step forward, You see a grey box 1 step right.
Action 8: left turn.
Observation 9: You carry a grey key, You see a wall 3 steps left, You see a wall 2 steps right, You see a blue ball 2 steps left and 3 steps forward, You see a red box 2 steps left and 2 steps forward, You see a yellow ball 3 steps forward.
Action 9: move forward.
Observation 10: You carry a grey key, You see a wall 3 steps left, You see a wall 2 steps right, You see a blue ball 2 steps left and 2 steps forward, You see a red box 2 steps left and 1 step forward, You see a yellow ball 2 steps forward.
Action 10: drop.
Successfully finished the goal.

Expert Demonstration Trajectory 2:

Goal of the agent: put the grey box next to the grey key

Observation 1: You see a wall 4 steps forward, You see a green ball 1 step right and 3 steps forward, You see a grey key 2 steps right and 3 steps forward, You see a red box 3 steps right and 3 steps forward.
Action 1: right turn.
Observation 2: You see a wall 4 steps forward, You see a wall 1 step right, You see a red box 3 steps left and 3 steps forward, You see a grey key 3 steps left and 2 steps forward, You see a green ball 3 steps left and 1 step forward.
Action 2: left turn.
Observation 3: You see a wall 4 steps forward, You see a green ball 1 step right and 3 steps forward, You see a grey key 2 steps right and 3 steps forward, You see a red box 3 steps right and 3 steps forward.
Action 3: left turn.
Observation 4: You see a wall 1 step left, You see a blue key 5 steps forward, You see a yellow box 1 step right and 5 steps forward, You see a grey box 3 steps right and 4 steps forward.
Action 4: move forward.
Observation 5: You see a wall 1 step left, You see a blue key 4 steps forward, You see a yellow box 1 step right and 4 steps forward, You see a grey box 3 steps right and 3 steps forward.
Action 5: move forward.
Observation 6: You see a wall 1 step left, You see a blue key 3 steps forward, You see a yellow box 1 step right and 3 steps forward, You see a grey box 3 steps right and 2 steps forward.
Action 6: move forward.
Observation 7: You see a wall 1 step left, You see a blue key 2 steps forward, You see a yellow box 1 step right and 2 steps forward, You see a grey box 3 steps right and 1 step forward.
Action 7: move forward.
Observation 8: You see a wall 1 step left, You see a blue key 1 step forward, You see a yellow box 1 step right and 1 step forward, You see a grey box 3 steps right.
Action 8: right turn.
Observation 9: You see a yellow box 1 step left and 1 step forward, You see a blue key 1 step left, You see a grey box 3 steps forward.
Action 9: move forward.
Observation 10: You see a yellow box 1 step left, You see a grey box 2 steps forward.
Action 10: move forward.
Observation 11: You see a wall 2 steps left, You see a grey box 1 step forward.
Action 11: pick up.
Observation 12: You carry a grey box, You see a wall 2 steps forward, You see a wall 2 steps left.
Action 12: right turn.
Observation 13: You carry a grey box, You see a wall 2 steps left, You see a wall 3 steps right, You see a grey key 1 step left and 6 steps forward, You see a green ball 1 step left and 5 steps forward.
Action 13: move forward.
Observation 14: You carry a grey box, You see a wall 2 steps left, You see a wall 3 steps right, You see a red box 1 step left and 6 steps forward, You see a grey key 1 step left and 5 steps forward, You see a green ball 1 step left and 4 steps forward.
Action 14: move forward.
Observation 15: You carry a grey box, You see a wall 6 steps forward, You see a wall 2 steps left, You see a wall 3 steps right, You see a red box 1 step left and 5 steps forward, You see a grey key 1 step left and 4 steps forward, You see a green ball 1 step left and 3 steps forward.
Action 15: move forward.
Observation 16: You carry a grey box, You see a wall 5 steps forward, You see a wall 2 steps left, You see a wall 3 steps right, You see a red box 1 step left and 4 steps forward, You see a grey key 1 step left and 3 steps forward, You see a green ball 1 step left and 2 steps forward.
Action 16: move forward.
Observation 17: You carry a grey box, You see a wall 4 steps forward, You see a wall 2 steps left, You see a wall 3 steps right, You see a red box 1 step left and 3 steps forward, You see a grey key 1 step left and 2 steps forward, You see a green ball 1 step left and 1 step forward.
Action 17: move forward.
Observation 18: You carry a grey box, You see a wall 3 steps forward, You see a wall 3 steps right, You see a red box 1 step left and 2 steps forward, You see a grey key 1 step left and 1 step forward, You see a green ball 1 step left.
Action 18: drop.
Successfully finished the goal.

Expert Demonstration Trajectory 3:

Goal of the agent: put the green ball next to the yellow key

Observation 1: You see a wall 3 steps left, You see a wall 2 steps right, You see a green ball 2 steps left and 6 steps forward, You see a blue box 6 steps forward, You see a yellow key 1 step right and 2 steps forward.
Action 1: move forward.
Observation 2: You see a wall 3 steps left, You see a wall 2 steps right, You see a purple key 2 steps left and 6 steps forward, You see a green ball 2 steps left and 5 steps forward, You see a blue box 5 steps forward, You see a yellow key 1 step right and 1 step forward.
Action 2: move forward.
Observation 3: You see a wall 3 steps left, You see a purple key 2 steps left and 5 steps forward, You see a green ball 2 steps left and 4 steps forward, You see a blue box 4 steps forward, You see a yellow key 1 step right.
Action 3: move forward.
Observation 4: You see a wall 3 steps left, You see a wall 2 steps right, You see a purple key 2 steps left and 4 steps forward, You see a green ball 2 steps left and 3 steps forward, You see a blue box 3 steps forward.
Action 4: move forward.
Observation 5: You see a wall 3 steps left, You see a wall 2 steps right, You see a purple key 2 steps left and 3 steps forward, You see a green ball 2 steps left and 2 steps forward, You see a blue box 2 steps forward.
Action 5: move forward.
Observation 6: You see a wall 3 steps left, You see a wall 2 steps right, You see a purple key 2 steps left and 2 steps forward, You see a green ball 2 steps left and 1 step forward, You see a blue box 1 step forward.
Action 6: left turn.
Observation 7: You see a wall 3 steps forward, You see a green ball 1 step right and 2 steps forward, You see a blue box 1 step right, You see a purple key 2 steps right and 2 steps forward.
Action 7: move forward.
Observation 8: You see a wall 2 steps forward, You see a wall 3 steps right, You see a green ball 1 step right and 1 step forward, You see a purple key 2 steps right and 1 step forward.
Action 8: move forward.
Observation 9: You see a wall 1 step forward, You see a green ball 1 step right, You see a purple key 2 steps right.
Action 9: right turn.
Observation 10: You see a wall 1 step left, You see a purple key 2 steps forward, You see a green ball 1 step forward, You see a blue box 2 steps right and 1 step forward.
Action 10: pick up.
Observation 11: You carry a green ball, You see a wall 1 step left, You see a purple key 2 steps forward, You see a blue box 2 steps right and 1 step forward.
Action 11: right turn.
Observation 12: You carry a green ball, You see a wall 4 steps forward, You see a purple key 2 steps left, You see a blue box 1 step left and 2 steps forward, You see a yellow key 3 steps right and 3 steps forward.
Action 12: move forward.
Observation 13: You carry a green ball, You see a wall 3 steps forward, You see a wall 3 steps left, You see a blue box 1 step left and 1 step forward, You see a yellow key 3 steps right and 2 steps forward.
Action 13: move forward.
Observation 14: You carry a green ball, You see a wall 2 steps forward, You see a blue box 1 step left, You see a yellow key 3 steps right and 1 step forward.
Action 14: move forward.
Observation 15: You carry a green ball, You see a wall 1 step forward, You see a wall 3 steps left, You see a yellow key 3 steps right.
Action 15: right turn.
Observation 16: You carry a green ball, You see a wall 1 step left, You see a grey box 6 steps forward, You see a yellow key 3 steps forward, You see a yellow ball 3 steps right and 6 steps forward.
Action 16: move forward.
Observation 17: You carry a green ball, You see a wall 1 step left, You see a grey box 5 steps forward, You see a yellow key 2 steps forward, You see a yellow ball 3 steps right and 5 steps forward.
Action 17: drop.
Successfully finished the goal.

Expert Demonstration Trajectory 4:

Goal of the agent: put the yellow key next to the blue box

Observation 1: You see a wall 2 steps left, You see a wall 3 steps right, You see a grey key 1 step left and 2 steps forward, You see a blue key 2 steps forward, You see a blue box 1 step right and 3 steps forward.
Action 1: right turn.
Observation 2: You see a wall 3 steps forward, You see a blue box 3 steps left and 1 step forward, You see a blue key 2 steps left, You see a yellow key 2 steps right and 2 steps forward, You see a blue ball 3 steps right.
Action 2: move forward.
Observation 3: You see a wall 2 steps forward, You see a blue box 3 steps left, You see a yellow key 2 steps right and 1 step forward.
Action 3: move forward.
Observation 4: You see a wall 1 step forward, You see a yellow key 2 steps right.
Action 4: right turn.
Observation 5: You see a wall 1 step left, You see a yellow key 2 steps forward, You see a purple ball 1 step right and 4 steps forward, You see a blue ball 2 steps right and 3 steps forward.
Action 5: move forward.
Observation 6: You see a wall 1 step left, You see a yellow key 1 step forward, You see a purple ball 1 step right and 3 steps forward, You see a blue ball 2 steps right and 2 steps forward.
Action 6: pick up.
Observation 7: You carry a yellow key, You see a wall 5 steps forward, You see a wall 1 step left, You see a purple ball 1 step right and 3 steps forward, You see a blue ball 2 steps right and 2 steps forward.
Action 7: right turn.
Observation 8: You carry a yellow key, You see a wall 4 steps forward, You see a purple ball 3 steps left and 1 step forward, You see a blue ball 2 steps left and 2 steps forward, You see a grey key 3 steps right and 3 steps forward, You see a blue key 3 steps right and 2 steps forward.
Action 8: move forward.
Observation 9: You carry a yellow key, You see a wall 3 steps forward, You see a purple ball 3 steps left, You see a blue ball 2 steps left and 1 step forward, You see a grey key 3 steps right and 2 steps forward, You see a blue key 3 steps right and 1 step forward.
Action 9: right turn.
Observation 10: You carry a yellow key, You see a wall 3 steps left, You see a wall 2 steps right, You see a grey key 2 steps left and 3 steps forward, You see a blue key 1 step left and 3 steps forward, You see a blue box 4 steps forward.
Action 10: move forward.
Observation 11: You carry a yellow key, You see a wall 3 steps left, You see a wall 2 steps right, You see a grey key 2 steps left and 2 steps forward, You see a blue key 1 step left and 2 steps forward, You see a blue box 3 steps forward.
Action 11: move forward.
Observation 12: You carry a yellow key, You see a wall 3 steps left, You see a wall 2 steps right, You see a grey key 2 steps left and 1 step forward, You see a blue key 1 step left and 1 step forward, You see a blue box 2 steps forward.
Action 12: drop.
Successfully finished the goal.

Expert Demonstration Trajectory 5:

Goal of the agent: put the yellow key next to the red ball

Observation 1: You see a wall 3 steps forward, You see a wall 1 step right, You see a green ball 3 steps left and 1 step forward, You see a green box 2 steps left and 2 steps forward, You see a red ball 1 step left and 2 steps forward.
Action 1: left turn.
Observation 2: You see a wall 4 steps forward, You see a wall 3 steps right, You see a yellow ball 3 steps left and 2 steps forward, You see a green ball 1 step right and 3 steps forward, You see a green box 2 steps right and 2 steps forward, You see a red ball 2 steps right and 1 step forward.
Action 2: left turn.
Observation 3: You see a wall 1 step left, You see a yellow key 1 step right and 5 steps forward, You see a yellow ball 2 steps right and 3 steps forward, You see a purple key 3 steps right and 4 steps forward.
Action 3: move forward.
Observation 4: You see a wall 6 steps forward, You see a wall 1 step left, You see a yellow key 1 step right and 4 steps forward, You see a yellow ball 2 steps right and 2 steps forward, You see a purple key 3 steps right and 3 steps forward.
Action 4: move forward.
Observation 5: You see a wall 5 steps forward, You see a wall 1 step left, You see a yellow key 1 step right and 3 steps forward, You see a yellow ball 2 steps right and 1 step forward, You see a purple key 3 steps right and 2 steps forward.
Action 5: move forward.
Observation 6: You see a wall 4 steps forward, You see a wall 1 step left, You see a yellow key 1 step right and 2 steps forward, You see a yellow ball 2 steps right, You see a purple key 3 steps right and 1 step forward.
Action 6: move forward.
Observation 7: You see a wall 3 steps forward, You see a wall 1 step left, You see a yellow key 1 step right and 1 step forward, You see a purple key 3 steps right.
Action 7: move forward.
Observation 8: You see a wall 2 steps forward, You see a wall 1 step left, You see a yellow key 1 step right.
Action 8: right turn.
Observation 9: You see a wall 2 steps left, You see a yellow key 1 step forward, You see a purple key 1 step right and 3 steps forward, You see a yellow ball 2 steps right and 2 steps forward.
Action 9: pick up.
Observation 10: You carry a yellow key, You see a wall 4 steps forward, You see a wall 2 steps left, You see a purple key 1 step right and 3 steps forward, You see a yellow ball 2 steps right and 2 steps forward.
Action 10: move forward.
Observation 11: You carry a yellow key, You see a wall 3 steps forward, You see a wall 2 steps left, You see a purple key 1 step right and 2 steps forward, You see a yellow ball 2 steps right and 1 step forward.
Action 11: right turn.
Observation 12: You carry a yellow key, You see a wall 3 steps left, You see a wall 2 steps right, You see a green ball 2 steps left and 6 steps forward, You see a purple key 2 steps left and 1 step forward, You see a yellow ball 1 step left and 2 steps forward.
Action 12: move forward.
Observation 13: You carry a yellow key, You see a wall 2 steps right, You see a green ball 2 steps left and 5 steps forward, You see a purple key 2 steps left, You see a green box 1 step left and 6 steps forward, You see a yellow ball 1 step left and 1 step forward, You see a red ball 6 steps forward.
Action 13: move forward.
Observation 14: You carry a yellow key, You see a wall 2 steps right, You see a green ball 2 steps left and 4 steps forward, You see a green box 1 step left and 5 steps forward, You see a yellow ball 1 step left, You see a red ball 5 steps forward.
Action 14: move forward.
Observation 15: You carry a yellow key, You see a wall 3 steps left, You see a wall 2 steps right, You see a green ball 2 steps left and 3 steps forward, You see a green box 1 step left and 4 steps forward, You see a red ball 4 steps forward.
Action 15: move forward.
Observation 16: You carry a yellow key, You see a wall 3 steps left, You see a wall 2 steps right, You see a green ball 2 steps left and 2 steps forward, You see a green box 1 step left and 3 steps forward, You see a red ball 3 steps forward.
Action 16: move forward.
Observation 17: You carry a yellow key, You see a wall 3 steps left, You see a wall 2 steps right, You see a green ball 2 steps left and 1 step forward, You see a green box 1 step left and 2 steps forward, You see a red ball 2 steps forward.
Action 17: drop.
Successfully finished the goal.

Currently, as the reinforcement learning agent, you are following the following list of strategies when making action decisions in this 'Put Next' environment:\n\n'''


suffix = {}

suffix['MiniGrid-Dynamic-Obstacles-6x6-v0'] = '''Now upon analyzing the above observation-action pairs with low advantage values, and based on your analysis and understanding of the 5 expert demonstrations of oracle trajectories provided to you earlier, please modify and update the list of strategies that you are currently following if you are confident that it is appropriate to do so. You can correct existing strategy items if you think they are inaccurate, you can add new strategy items if you think they are currently missing, and you can delete existing strategy items if you think they are wrong. Please remember that the above advantage values are estimated by the value network of the PPO algorithm, and thus may not be entirely accurate and should be analyzed with caution. Therefore, you should consider the evidence suggested by the above observation-action pairs with low advantage values, the patterns and insights exhibited by the expert demonstration trajectories, and your own understanding, reasoning and judgement about this 'Dynamic Obstacles' task all together to make wise decisions when modifying and updating the list of strategies. Please only return the updated list of strategies without any other text before or after the list.'''

suffix['BabyAI-PutNextS6N3-v0'] = '''Now upon analyzing the above observation-action pairs with low advantage values, and based on your analysis and understanding of the 5 expert demonstrations of oracle trajectories provided to you earlier, please modify and update the list of strategies that you are currently following if you are confident that it is appropriate to do so. You can correct existing strategy items if you think they are inaccurate, you can add new strategy items if you think they are currently missing, and you can delete existing strategy items if you think they are wrong. Please remember that the above advantage values are estimated by the value network of the PPO algorithm, and thus may not be entirely accurate and should be analyzed with caution. Therefore, you should consider the evidence suggested by the above observation-action pairs with low advantage values, the patterns and insights exhibited by the expert demonstration trajectories, and your own understanding, reasoning and judgement about this 'Put Next' task all together to make wise decisions when modifying and updating the list of strategies. Please only return the updated list of strategies without any other text before or after the list.'''

def update_model_strategy(model, env_name, exps, num_pairs, api):
    advantage = exps.advantage.cpu()
    obs = exps.obs
    action = exps.action.cpu()

    sorted_advantage, indices = torch.sort(advantage)

    chosen_indices = list(indices.numpy())[:num_pairs]

    prompt = prefix[env_name] + model.strategies + '\n\nAnd in your current iteration of experience collection during a PPO training process, the following ' + str(num_pairs) + ''' observation-action pairs received the lowest advantage values, which indicates that these action decisions might not be optimal:\n\n'''

    all_pairs = ''

    for i in range(num_pairs):
        all_pairs += 'Pair ' + str(i+1) + ':\n'
        text = gen_text_desc(obs.image[chosen_indices[i],:,:,:].cpu().numpy())
        all_pairs += 'Observation: ' + ', '.join(text['descriptions']) + '.\n'
        all_pairs += 'Action: ' + ID_TO_ACTION[action.numpy()[chosen_indices[i]]] + '.\n'
        all_pairs += 'Advantage value: ' + str(advantage[chosen_indices[i]].numpy()) + '\n\n'

    prompt += all_pairs

    prompt += suffix[env_name]

    response = client.chat.completions.create(
        model=api,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    new_strategies = response.choices[0].message.content

    model.strategies = new_strategies

    return all_pairs, new_strategies
