import argparse
import sys
import random
import os
from tqdm import tqdm
import pickle
import blosc
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
import tensorboardX

# Add the parent directory of 'script' to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import utils
from utils import device
from utils.obs2text import gen_text_desc
from models.model import ACModel
from scripts.model_evaluation import group_evaluate

# Parse arguments

parser = argparse.ArgumentParser()

parser.add_argument("--use-strategy", action="store_true",
                    help="whether to incorporate the strategy module into the RL agent")
parser.add_argument("--use-lora", action="store_true",
                    help="whether to use LORA (low-rank adaptation) on the language model during finetuning")
parser.add_argument("--lora-rank", type=int, default=None,
                    help="rank r for LORA finetuning")
parser.add_argument("--freeze-lm", action="store_true",
                    help="whether to freeze the underlying transformer language model except for its language modeling head (LM-head) during training")
parser.add_argument("--argmax", action="store_true",
                    help="whether to only use the action with maximum probability during evaluation")
parser.add_argument("--env-name", type=str, default='MiniGrid-Dynamic-Obstacles-6x6-v0',
                    help="Name of the BabyAI environment")
parser.add_argument("--model-name", type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                    help="Name of the core language model")
parser.add_argument("--optimizer-name", type=str, default='Adam',
                    help="Name of the optimizer")
parser.add_argument("--scheduler-type", type=str, default='None',
                    help="Type of the learning rate scheduler")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--epochs", type=int, default=50,
                    help="Number of epochs for the behavioral cloning training")
parser.add_argument("--batch-size", type=int, default=4,
                    help="batch size for behavioral cloning")
parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                    help="batch size for behavioral cloning")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate")
parser.add_argument("--epsilon", type=float, default=1e-5,
                    help="epsilon value of the Adam/AdamW optimizer")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--num-demo-train", type=int, default=10,
                    help="Number of sampled expert demo trajectories for the training of behavioral cloning")
parser.add_argument("--num-demo-test", type=int, default=10,
                    help="Number of sampled expert demo trajectories for the testing of behavioral cloning")
parser.add_argument("--num-eval", type=int, default=20,
                    help="Number of evaluatation episodes")
parser.add_argument("--value-coeff", type=float, default=0.5,
                    help="the weight ratio of the value loss term with respect to the policy loss term in the loss function for behavioral cloning")
parser.add_argument("--entropy-coeff", type=float, default=0.001,
                    help="the weight ratio of the entropy loss term with respect to the policy loss term in the loss function for behavioral cloning")
parser.add_argument("--save", action="store_true",
                    help="whether to save the model")
parser.add_argument("--save-every", type=int, default=10,
                    help="every number of epochs to save the model")

args = parser.parse_args()


header_dict = {}

header_dict['MiniGrid-Dynamic-Obstacles-6x6-v0'] = {}
header_dict['BabyAI-PutNextS6N3-v0'] = {}

header_dict['MiniGrid-Dynamic-Obstacles-6x6-v0'][True] = '''Possible action of the agent: left turn, right turn, move forward.\n\n'''

header_dict['MiniGrid-Dynamic-Obstacles-6x6-v0'][False] = 'Possible action of the agent: left turn, right turn, move forward.\n\n'

header_dict['BabyAI-PutNextS6N3-v0'][True] = '''Possible actions of the agent: left turn, right turn, move forward, pick up, drop.\n\n'''

header_dict['BabyAI-PutNextS6N3-v0'][False] = 'Possible actions of the agent: left turn, right turn, move forward, pick up, drop.\n\n'


strategies_dict = {}

strategies_dict['MiniGrid-Dynamic-Obstacles-6x6-v0'] = {}
strategies_dict['MiniGrid-Dynamic-Obstacles-6x6-v0'][True] = '''1. Avoiding Obstacles and Walls:
    - The agent should turn left or right when it observes a blue ball (obstacle) or a wall directly in the forward path (e.g., when a wall or a blue ball is 1 step forward).
    - The agent should move forward when no immediate obstacles or walls are in the directly forward path, even if obstacles are present in adjacent squares.
2. Navigating Towards the Goal:
    - When the green goal is observed in a direction with no immediate obstacles in between, the agent should orient itself towards that direction and move forward.
    - The agent should prioritize moving in a direction where the distance to the goal decreases, provided it does not lead to a collision with obstacles.
3. Orienting Towards the Goal:
    - If the goal is observed to the left or right, the agent should turn left or right, respectively, to face the goal more directly, especially when the path ahead is clear.
    - The agent should adjust its orientation (turn left or right) to ensure that the goal remains in its field of view as it navigates around obstacles.
4. Path Correction and Obstacle Avoidance:
    - Upon encountering an obstacle close by (e.g., 1 step left or right), the agent often turns away from the obstacle if the goal is not in the immediate path of the obstacle.
    - If the agent is adjacent to a wall or near an obstacle, and the goal is not directly ahead, it might turn away from the wall/obstacle to explore alternative paths.
5. Advanced Maneuvering:
    - In situations where the agent is surrounded by obstacles on multiple sides but has a clear path to the goal in one direction, it chooses to move towards the clear path.
    - The agent demonstrates a pattern of 'reorientation' by turning left or right, followed by a forward movement when the path directly to the goal is initially obstructed but becomes clear after reorientation.
6. Final Approach to the Goal:
    - When the agent is close to the goal (e.g., the goal is 1 or 2 steps forward), it prioritizes moving forward directly towards the goal, even if there are obstacles nearby but not directly in the path.
7. Efficient Use of Space:
    - The agent seems to leverage the entire available space, avoiding corners and dead-ends where maneuverability is limited unless the goal is located in such areas.'''
strategies_dict['MiniGrid-Dynamic-Obstacles-6x6-v0'][False] = ''

strategies_dict['BabyAI-PutNextS6N3-v0'] = {}
strategies_dict['BabyAI-PutNextS6N3-v0'][True] = '''1. Strategy for Object Acquisition:
    - The agent should pick up the target object as soon as it is within one step forward.
    - The agent should move forward towards the target object until it is directly in front of the agent, then execute the pick-up action.
2. Turning Strategy for Navigation:
    - The agent should turn right or left to face the direction where the target object or destination is most visible or nearest in terms of steps forward.
3. Long Distance Navigation:
    - If the target object is visible but several steps away, the agent should move forward continuously until reaching the object, adjusting the direction if necessary through minimal turning.
4. Obstacle Avoidance and Path Correction:
    - When an obstacle (like a wall) is one step forward, the agent should turn left or right based on the direction that leads towards the target object or provides clear path forward.
5. Placement Strategy for Object:
    - Once the target object is picked up, the agent should navigate towards the second designated object (goal object).
    - The agent should position itself so that dropping the carried object will place it next to the goal object. This might involve positioning the agent either directly adjacent or a single move away from where the object needs to be placed.
6. Final Actions Before Placement:
    - Before placing the object next to the goal object, if the agent is not correctly aligned with the drop location, it should adjust by turning towards the appropriate direction where a drop action would successfully place the object next to the goal object.
    - The agent should drop the carried object when the target placement is one step to the left or right of the current forward path.
7. Efficient Turning and Movement:
    - The agent should minimize the number of turns by planning a path that requires the fewest directional changes.
    - Upon reaching a dead-end or facing an immediate obstacle, the agent should turn towards the open path that directly progresses towards the goal.
8. Use of Environmental Cues:
    - The agent should use visible cues from the environment, such as the position of walls and other objects, to infer the least obstructed path towards the target or goal objects.
9. Optimal Path Finding:
    - The agent should identify and follow the shortest path to the target object, considering both distance and obstacles, ensuring that each step taken maximizes the closeness to the goal.'''
strategies_dict['BabyAI-PutNextS6N3-v0'][False] = ''

strategy_label = {True: 'strategy', False: 'no-strategy'}
lora_label = {True: 'lora', False: 'no-lora'}

num_actions_dict = {'MiniGrid-Dynamic-Obstacles-6x6-v0': 3, 'BabyAI-PutNextS6N3-v0': 5}

train_demo_path = '../expert-data/' + args.env_name + '_agent_' + str(100) + '.pkl'

with open(train_demo_path, 'rb') as f:
    all_train_demos = pickle.load(f)

all_train_demos = all_train_demos[:args.num_demo_train]
train_demos = all_train_demos

test_demo_path = '../expert-data/' + args.env_name + '_agent_valid_' + str(args.num_demo_test) + '.pkl'

with open(test_demo_path, 'rb') as f:
    all_test_demos = pickle.load(f)

test_demos = all_test_demos

def prepare_data(demos):

    data = []

    for trajectory in tqdm(demos):

        num_steps = len(trajectory[3])

        final_reward = trajectory[5]

        values = [final_reward]

        for i in range(num_steps-1):
            values.append(values[-1] * args.discount)

        values.reverse()

        for i in range(num_steps):

            description = header_dict[args.env_name][args.use_strategy]

            if args.use_strategy:
                description += 'Rules to follow:\n' + strategies_dict[args.env_name][args.use_strategy] + '\n\n'

            description += 'Goal of the agent: ' + trajectory[0] + '.\n'

            description += 'Observation: ' + ', '.join(gen_text_desc(blosc.unpack_array(trajectory[1])[i])['descriptions']) + '.\n'
            description += 'Action:'

            action = trajectory[3][i].value
            value = values[i]

            data.append({'description': description, 'action': action, 'value': value})

    return data


train_data = prepare_data(train_demos)
test_data = prepare_data(test_demos)


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        description = item['description']
        action = torch.tensor(item['action'], dtype=torch.int64)
        value = torch.tensor(item['value'], dtype=torch.float)
        return description, action, value

train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

acmodel = ACModel(core_model_name = args.model_name, header = header_dict[args.env_name][args.use_strategy], strategies = strategies_dict[args.env_name][args.use_strategy], num_actions = num_actions_dict[args.env_name], use_lora = args.use_lora, lora_rank = args.lora_rank, freeze_lm = args.freeze_lm, use_strategy = args.use_strategy, output_attentions = False, use_tensor = False)

if args.optimizer_name == 'AdamW':
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, acmodel.parameters()), lr = args.lr, eps = args.epsilon)
elif args.optimizer_name == 'Adam':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, acmodel.parameters()), lr = args.lr, eps = args.epsilon)
else:
    raise ValueError("Can't recognize the specified optimizer type.")

if args.scheduler_type == 'None':
    pass
elif args.scheduler_type == 'Cosine':
    scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer, num_warmup_steps = 0, num_training_steps = 100)
else:
    raise ValueError("Can't recognize the specified type of learning rate scheduler.")

acmodel.to(device)
acmodel.set_device()
acmodel.train()

tensorboard_path = os.path.join('tb-bc', args.env_name, args.model_name, 'lm-head', args.optimizer_name, args.scheduler_type, 'lr_' + str(args.lr), 'eps_' + str(args.epsilon), 'entropy-' + str(args.entropy_coeff), 'batch-size-' + str(args.batch_size), 'gc-' + str(args.gradient_accumulation_steps), 'BC-' + str(args.num_demo_train), strategy_label[args.use_strategy], lora_label[args.use_lora], 'r-' + str(args.lora_rank))

tb_writer = tensorboardX.SummaryWriter(tensorboard_path)

training_frames = 0

max_average_return = 0

for epoch in range(args.epochs):

    acmodel.train()

    optimizer.zero_grad()

    for i, (descriptions, oracle_actions, oracle_values) in enumerate(tqdm(train_dataloader, desc='Training Epoch')):

        dist, value = acmodel(obs = None, h2obs = None, h2a = None, h1obs = None, h1a = None, ready_descriptions = list(descriptions))

        entropy = dist.entropy().mean()

        oracle_actions, oracle_values = oracle_actions.to(device), oracle_values.to(device)

        policy_loss = F.cross_entropy(dist.logits, oracle_actions)

        loss = policy_loss - args.entropy_coeff * dist.entropy().mean()

        tb_writer.add_scalar('Train/Overall_Loss', loss.item(), training_frames)
        tb_writer.add_scalar('Train/Policy_Loss', policy_loss.item(), training_frames)
        tb_writer.add_scalar('Train/Entropy', entropy.item(), training_frames)

        loss = loss / args.gradient_accumulation_steps

        loss.backward()

        if (i + 1) % args.gradient_accumulation_steps == 0 or i + 1 == len(train_dataloader):

            torch.nn.utils.clip_grad_norm_(acmodel.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        training_frames += len(descriptions)

    if args.scheduler_type != 'None':
        scheduler.step()

    print(training_frames)

    # Validation
        
    loss_sum = 0
    policy_loss_sum = 0
    value_loss_sum = 0

    acmodel.eval()

    for descriptions, oracle_actions, oracle_values in tqdm(test_dataloader, desc = 'Validation Stage'):

        with torch.no_grad():

            dist, value = acmodel(obs = None, h2obs = None, h2a = None, h1obs = None, h1a = None, ready_descriptions = list(descriptions))

        oracle_actions, oracle_values = oracle_actions.to(device), oracle_values.to(device)

        policy_loss = F.cross_entropy(dist.logits, oracle_actions, reduction = 'sum')

        loss = policy_loss - args.entropy_coeff * dist.entropy().mean()

        loss_sum += loss.item()
        policy_loss_sum += policy_loss.item()

    tb_writer.add_scalar('Validation/Overall_Loss', loss_sum / len(test_dataset), training_frames)
    tb_writer.add_scalar('Validation/Policy_Loss', policy_loss_sum / len(test_dataset), training_frames)

    validation_loss = loss_sum / len(test_dataset)
    print(validation_loss)

    # Evaluate:

    print('Epoch: ', epoch)
    print(acmodel.use_strategy)
    print(acmodel.strategies)
    
    average_return, success_rate = group_evaluate(acmodel, env_name = args.env_name, num_episodes = args.num_eval, argmax = args.argmax, max_steps = 60)

    tb_writer.add_scalar('Evaluation/Average_Return', average_return, training_frames)
    tb_writer.add_scalar('Evaluation/Success_Rate', success_rate, training_frames)
    
    # Save:

    model_save_path = os.path.join('saved_models', args.env_name, args.model_name, 'BC-' + str(args.num_demo_train), strategy_label[args.use_strategy], lora_label[args.use_lora], 'r-' + str(args.lora_rank), 'epoch-' + str(epoch+1))

    acmodel.text_model.save_pretrained(model_save_path)
    torch.save(acmodel.critic.state_dict(), os.path.join(model_save_path, 'critic.pt'))
    