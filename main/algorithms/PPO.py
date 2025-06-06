import torch
import numpy
import copy
from algorithms.algo_utils.penv import ParallelEnv
from algorithms.algo_utils.dictlist import DictList
from tqdm import tqdm

psuedo_image = numpy.full((7, 7, 3), -1)
psuedo_action = -1

class PPO:
    """The class for the vanilla PPO algorithm (baseline)."""

    def __init__(self, envs, acmodel, optimizer_name, device = None, num_frames_per_proc = 128, discount = 0.99, lr = 0.00001, gae_lambda = 0.95, entropy_coef = 0.01, value_loss_coef = 0.5, max_grad_norm = 0.5, recurrence = 1, adam_eps=1e-8, clip_eps = 0.2, epochs = 4, batch_size = 2, gradient_accumulation_steps = 16, reshape_reward = None, anneal = False):

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.min_entropy_coef = self.entropy_coef * 0.01
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.reshape_reward = reshape_reward
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.anneal = anneal

        assert self.batch_size % self.recurrence == 0

        if optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.acmodel.parameters()), lr = self.lr, eps = adam_eps)
        elif optimizer_name == 'Adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.acmodel.parameters()), lr = self.lr, eps = adam_eps)
        else:
            raise ValueError("Can't recognize the specified optimizer type.")

        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=20)

        self.batch_num = 0

        def preprocess_images(images, device=None):
            images = numpy.array(images)
            return torch.tensor(images, device=device, dtype=torch.float)

        def preprocess_obss(obss, device=None):
            return DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
                "mission": numpy.array([obs['mission'] for obs in obss]),
                "direction": numpy.array([obs['direction'] for obs in obss])
            })
        
        self.preprocess_obss = preprocess_obss

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        self.h1obs = None
        self.h1obss = [None] * (shape[0])
        self.h1actions = torch.zeros(*shape, device=self.device, dtype=torch.int)

        self.h2obs = None
        self.h2obss = [None] * (shape[0])
        self.h2actions = torch.zeros(*shape, device=self.device, dtype=torch.int)

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages."""

        self.acmodel.train()

        print('Collecting experiences:')

        for i in tqdm(range(self.num_frames_per_proc)):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            if i == 0:
                self.h1obs = copy.deepcopy(self.obs)
                for j in range(len(self.h1obs)):
                    self.h1obs[j]['image'] = psuedo_image

                self.h2obs = copy.deepcopy(self.h1obs)

                h1a = torch.full((len(self.obs),), -1)
                h2a = torch.full((len(self.obs),), -1)
            elif i == 1:
                self.h2obs = copy.deepcopy(self.obs)
                for j in range(len(self.h2obs)):
                    self.h2obs[j]['image'] = psuedo_image
                h2a = torch.full((len(self.obs),), -1)

                self.h1obs = copy.deepcopy(self.obss[0])
                h1a = copy.deepcopy(self.actions[0])
            else:
                self.h2obs = copy.deepcopy(self.obss[i-2])
                self.h1obs = copy.deepcopy(self.obss[i-1])
                h2a = copy.deepcopy(self.actions[i-2])
                h1a = copy.deepcopy(self.actions[i-1])

                for j in range(len(self.h1obs)):
                    if self.masks[i-1][j] == 0:
                        self.h1obs[j]['image'] = psuedo_image
                        h1a[j] = psuedo_action
                        self.h2obs[j]['image'] = psuedo_image
                        h2a[j] = psuedo_action
                    elif self.masks[i-2][j] == 0:
                        self.h2obs[j]['image'] = psuedo_image
                        h2a[j] = psuedo_action

            preprocessed_h1obs = self.preprocess_obss(self.h1obs, device = self.device)
            preprocessed_h2obs = self.preprocess_obss(self.h2obs, device = self.device)

            with torch.no_grad():
                dist, value = self.acmodel(preprocessed_obs, preprocessed_h2obs, h2a, preprocessed_h1obs, h1a)

            action = dist.sample()

            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = tuple(a | b for a, b in zip(terminated, truncated))

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action

            self.h1obss[i] = self.h1obs
            self.h2obss[i] = self.h2obs

            self.h1actions[i] = h1a
            self.h2actions[i] = h2a

            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences
            
        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

        preprocessed_h1obs = self.preprocess_obss(self.h1obs, device = self.device)
        preprocessed_h2obs = self.preprocess_obss(self.h2obs, device = self.device)

        with torch.no_grad():
            _, next_value = self.acmodel(preprocessed_obs, preprocessed_h2obs, h2a, preprocessed_h1obs, h1a)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        
        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        
        exps.h1obs = [self.h1obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        
        exps.h2obs = [self.h2obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)

        exps.h1action = self.h1actions.transpose(0, 1).reshape(-1)
        exps.h2action = self.h2actions.transpose(0, 1).reshape(-1)

        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        exps.h1obs = self.preprocess_obss(exps.h1obs, device=self.device)
        exps.h2obs = self.preprocess_obss(exps.h2obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs


    def update_parameters(self, exps):

        self.acmodel.train()

        print('Training Epochs:')
        for _ in tqdm(range(self.epochs)):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            print('Training Batches:')
            for index, inds in enumerate(tqdm(self._get_batches_starting_indexes())):
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    dist, value = self.acmodel(sb.obs, sb.h2obs, sb.h2action, sb.h1obs, sb.h1action)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                batch_loss.backward()

                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters() if p.grad is not None) ** 0.5

                if (index + 1) % self.gradient_accumulation_steps == 0 or index + 1 == len(self._get_batches_starting_indexes()):

                    torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()


                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        if self.anneal:

            self.scheduler.step()

            if self.entropy_coef > self.min_entropy_coef:
                self.entropy_coef = self.entropy_coef * 0.8

        return logs
    
    
    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
