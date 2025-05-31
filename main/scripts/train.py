import argparse
import time
import datetime
import os
import tensorboardX
import sys
import torch
import random
from peft import AutoPeftModelForCausalLM

# Add the parent directory of 'script' to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import utils
from utils import device
from models.model import ACModel
from algorithms.PPO import PPO
from scripts.model_evaluation import group_evaluate
from update_strategy import update_model_strategy

# Parse arguments

parser = argparse.ArgumentParser()

# General parameters

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
parser.add_argument("--max-steps", type=int, default=None,
                    help="set max steps for the env")
parser.add_argument("--anneal", action="store_true",
                    help="whether to perform annealing on key hyperparameters")
parser.add_argument("--use-distill", action="store_true",
                    help="whether to enable dynamic strategy induction")
parser.add_argument("--api", type=str, default='gpt-4o',
                    help="Name of the LLM API for distill queries")
parser.add_argument("--num-pairs", type=int, default=10,
                    help="number of (state, action) pairs with lowest advantage values to be considered for dynamic strategy induction during each epoch")
parser.add_argument("--warmup-epochs", type=int, default=0,
                    help="number of warm-up epochs before dynamic strategy induction is turned on")
parser.add_argument("--load-checkpoint", type=int, default=None,
                    help="the id of the BC checkpoint to load (None means no load).")
parser.add_argument("--num-demos", type=int, default=None,
                    help="number of expert demonstration trajectories (both during BC and during PPO)")
parser.add_argument("--env", type = str, default = 'MiniGrid-Dynamic-Obstacles-6x6-v0',
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model-name", type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', 
                    help="Name of the core language model")
parser.add_argument("--optimizer-name", type=str, default='Adam',
                    help="Name of the optimizer")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--dont-save-tb", action="store_true",
                    help="whether not to save training progress into the tensorboard file")
parser.add_argument("--procs", type=int, default=4,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--num-valid", type=int, default=30,
                    help="Number of validation episodes")
parser.add_argument("--num-comp", type=int, default=20,
                    help="Number of comparison episodes")

# Parameters for main algorithm

parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=64,
                    help="batch size for PPO (default: 128)")
parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                    help="gradient accumulation steps")
parser.add_argument("--frames-per-proc", type=int, default=64, 
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=1e-2,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.1,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated")

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


ID_TO_ACTION = {0: 'left turn', 1: 'right turn', 2: 'move forward', 3: 'pick up', 4: 'drop', 5: 'toggle'}

strategy_label = {True: 'strategy', False: 'no-strategy'}
lora_label = {True: 'lora', False: 'no-lora'}
anneal_label = {True: 'anneal', False: 'no-anneal'}
distill_label = {True: 'distill', False: 'no-distill'}

num_actions_dict = {'MiniGrid-Dynamic-Obstacles-6x6-v0': 3, 'BabyAI-PutNextS6N3-v0': 5}

if __name__ == "__main__":
    args = parser.parse_args()

    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_PPO_seed{args.seed}_{date}"

    model_name = args.model_name or default_model_name
    model_dir = utils.get_model_dir(model_name)

    tensorboard_path = os.path.join('ppo-results', args.env, args.model_name, 'n_demos-' + str(args.num_demos), 'cpid-' + str(args.load_checkpoint), args.optimizer_name, 'n_procs-' + str(args.procs), 'fpp-' + str(args.frames_per_proc), 'n_epochs-' + str(args.epochs), 'entropy-' + str(args.entropy_coef), 'gae-' + str(args.gae_lambda), 'clip-' + str(args.clip_eps), 'lr_' + str(args.lr), 'eps_' + str(args.optim_eps), 'batch-size-' + str(args.batch_size), 'gc-' + str(args.gradient_accumulation_steps), anneal_label[args.anneal], strategy_label[args.use_strategy], lora_label[args.use_lora], 'r-' + str(args.lora_rank), distill_label[args.use_distill], 'n_pairs-' + str(args.num_pairs), 'n_warmup-' + str(args.warmup_epochs), args.api)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(tensorboard_path)
    csv_file, csv_logger = utils.get_csv_logger(tensorboard_path)

    if not args.dont_save_tb:

        tb_writer = tensorboardX.SummaryWriter(tensorboard_path)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i, max_steps = args.max_steps))
    txt_logger.info("Environments loaded\n")

    # Load training status

    status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load model

    acmodel = ACModel(core_model_name = args.model_name, header = header_dict[args.env][args.use_strategy], strategies = strategies_dict[args.env][args.use_strategy], num_actions = num_actions_dict[args.env], use_lora = args.use_lora, lora_rank = args.lora_rank, freeze_lm = args.freeze_lm, use_strategy = args.use_strategy, output_attentions = False, use_tensor = False)

    if args.load_checkpoint is not None:

        model_path = os.path.join('../algorithms/saved_models', args.env, args.model_name, 'BC-' + str(args.num_demos), strategy_label[args.use_strategy], lora_label[args.use_lora], 'r-' + str(args.lora_rank), 'epoch-' + str(args.load_checkpoint))

        acmodel.text_model = AutoPeftModelForCausalLM.from_pretrained(model_path)

        acmodel.text_model.config.pad_token_id = acmodel.tokenizer.pad_token_id

        critic_state_dict = torch.load(os.path.join(model_path, 'critic.pt'))

        acmodel.critic.load_state_dict(critic_state_dict)

    acmodel.to(device)
    acmodel.set_device()

    txt_logger.info("Model loaded\n")

    # Load algorithm

    algo = PPO(envs, acmodel, args.optimizer_name, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence, args.optim_eps, args.clip_eps, args.epochs, args.batch_size, args.gradient_accumulation_steps, None, args.anneal)

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    # First evaluate the loaded model checkpoint

    validation_seeds = [10000 * (x + 1) for x in range(args.num_valid)] # validation set

    average_return, success_rate = group_evaluate(acmodel, env_name = args.env, all_seeds = validation_seeds, argmax = args.argmax, max_steps = args.max_steps)

    max_score = average_return

    tb_writer.add_scalar('Evaluation/Average_Return', average_return, num_frames)
    tb_writer.add_scalar('Evaluation/Success_Rate', success_rate, num_frames)
    tb_writer.add_scalar('Evaluation/New_Max_Score', max_score, num_frames)

    while num_frames < args.frames:

        acmodel.train()

        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()

        # save current model parameter into buffer (prepare for future retrieval)

        if args.use_distill:

            acmodel.text_model.save_pretrained(os.path.join(tensorboard_path, 'buffer_original'))
            torch.save(acmodel.critic.state_dict(), os.path.join(tensorboard_path, 'buffer_original', 'critic.pt'))

        logs2 = algo.update_parameters(exps)

        logs = {**logs1, **logs2}
        update_end_time = time.time()

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))
            
            txt_logger.info("{}\n".format(" ".join(sys.argv)))
            txt_logger.info("{}\n".format(args))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            if not args.dont_save_tb:

                for field, value in zip(header, data):
                    tb_writer.add_scalar(field, value, num_frames)


        num_frames += logs["num_frames"]
        update += 1

        # Update Model Strategies

        if args.use_distill:

            # Evaluate Model 1:

            random_seeds = random.sample(range(10000, 20000), args.num_comp)

            acmodel.eval()

            average_return_1, success_rate_1 = group_evaluate(acmodel, env_name = args.env, all_seeds = random_seeds, argmax = args.argmax, max_steps = args.max_steps)

            score_1 = average_return_1

            acmodel.train()

            # Save current model parameters into buffer as model_1:

            acmodel.text_model.save_pretrained(os.path.join(tensorboard_path, 'buffer_model_1'))
            torch.save(acmodel.critic.state_dict(), os.path.join(tensorboard_path, 'buffer_model_1', 'critic.pt'))

            current_strategies = acmodel.strategies

            # Load original model parameters from buffer_original:

            acmodel.text_model = AutoPeftModelForCausalLM.from_pretrained(os.path.join(tensorboard_path, 'buffer_original'))

            acmodel.text_model.config.pad_token_id = acmodel.tokenizer.pad_token_id

            critic_state_dict = torch.load(os.path.join(tensorboard_path, 'buffer_original', 'critic.pt'))

            acmodel.critic.load_state_dict(critic_state_dict)

            acmodel.to(device)
            acmodel.set_device()
            acmodel.train()

            # Perform DISTILL query

            all_pairs, new_strategies = update_model_strategy(acmodel, args.env, exps, args.num_pairs, api = args.api)

            print('New Strategies:')
            print(new_strategies)

            logs3 = algo.update_parameters(exps)

            acmodel.eval()

            average_return_2, success_rate_2 = group_evaluate(acmodel, env_name = args.env, all_seeds = random_seeds, argmax = args.argmax, max_steps = args.max_steps)

            score_2 = average_return_2

            acmodel.train()

            if score_2 < score_1:

                # Load buffer_model_1:

                acmodel.text_model = AutoPeftModelForCausalLM.from_pretrained(os.path.join(tensorboard_path, 'buffer_model_1'))

                acmodel.text_model.config.pad_token_id = acmodel.tokenizer.pad_token_id

                critic_state_dict = torch.load(os.path.join(tensorboard_path, 'buffer_model_1', 'critic.pt'))

                acmodel.critic.load_state_dict(critic_state_dict)

                # Restore strategies

                acmodel.strategies = current_strategies

                acmodel.to(device)
                acmodel.set_device()
                acmodel.train()

                tb_writer.add_scalar('Evaluation/Model_Choice', 1, num_frames)

                print('Model 1 wins!')

            else:

                tb_writer.add_scalar('Evaluation/Model_Choice', 2, num_frames)

                with open(os.path.join(tensorboard_path, 'distill_record.txt'), 'a') as file:
                    file.write('Epoch ' + str(update) + ':\n\n\n')
                    file.write('State-Action Pairs with Low Advantage Values:\n\n')
                    file.write(all_pairs + '\n')
                    file.write('New Strategies:\n\n')
                    file.write(new_strategies + '\n\n\n')

                print('Model 2 wins!\n')

        # Run overall evaluation

        acmodel.eval()

        validation_seeds = [10000 * (x + 1) for x in range(args.num_valid)] # validation set

        average_return, success_rate = group_evaluate(acmodel, env_name = args.env, all_seeds = validation_seeds, argmax = args.argmax, max_steps = args.max_steps)

        tb_writer.add_scalar('Evaluation/Average_Return', average_return, num_frames)
        tb_writer.add_scalar('Evaluation/Success_Rate', success_rate, num_frames)

        if average_return > max_score:
            max_score = average_return

            acmodel.text_model.save_pretrained(os.path.join(tensorboard_path, 'best_model'))
            torch.save(acmodel.critic.state_dict(), os.path.join(tensorboard_path, 'best_model', 'critic.pt'))

            with open(os.path.join(tensorboard_path, 'best_model', 'best_strategies.txt'), 'w') as file:
                file.write(acmodel.strategies)
        
        tb_writer.add_scalar('Evaluation/New_Max_Score', max_score, num_frames)

        acmodel.train()  
