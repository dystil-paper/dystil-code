import argparse
import time
import torch
import numpy
import math
import copy
from tqdm import tqdm
from algorithms.algo_utils.penv import ParallelEnv
from algorithms.algo_utils.dictlist import DictList

import utils

psuedo_image = numpy.full((7, 7, 3), -1)
psuedo_action = -1

def preprocess_images(images, device=None):
            images = numpy.array(images)
            return torch.tensor(images, device=device, dtype=torch.float)

def preprocess_obss(obss, device=None):
    return DictList({
        "image": preprocess_images([obs["image"] for obs in obss], device=device),
        "mission": numpy.array([obs['mission'] for obs in obss]),
        "direction": numpy.array([obs['direction'] for obs in obss])
    })

def evaluate(model, env_name, seeds, argmax = False, max_steps = None):

    procs = len(seeds)

    model.eval()

    device = model.device

    # Load environments

    envs = []
    for i in range(procs):
        env = utils.make_env(env_name, seeds[i], max_steps = max_steps)
        envs.append(env)
    env = ParallelEnv(envs)

    print("Environments loaded\n")

    max_steps = 1000

    max_frames_per_process = max_steps

    # Initialize experience containers

    shape = (max_frames_per_process, procs)

    obs = env.reset()
    obss = [None] * (shape[0])
    mask = torch.ones(shape[1], device=device)
    masks = torch.zeros(*shape, device=device)
    actions = torch.zeros(*shape, device=device, dtype=torch.int)
    rewards = torch.zeros(*shape, device=device)

    h1obs = None
    h1obss = [None] * (shape[0])
    h1actions = torch.zeros(*shape, device=device, dtype=torch.int)

    h2obs = None
    h2obss = [None] * (shape[0])
    h2actions = torch.zeros(*shape, device=device, dtype=torch.int)

    # Initialize log values

    log_episode_return = torch.zeros(procs, device=device)
    log_episode_num_frames = torch.zeros(procs, device=device)

    pbar = tqdm(total = 300, desc = 'Evaluation Episodes')

    frame_idx = 0

    all_returns = [None] * procs

    while None in all_returns:

        if frame_idx == 0:
            h1obs = copy.deepcopy(obs)
            for j in range(len(h1obs)):
                h1obs[j]['image'] = psuedo_image

            h2obs = copy.deepcopy(h1obs)

            h1a = torch.full((len(obs),), -1)
            h2a = torch.full((len(obs),), -1)
        elif frame_idx == 1:
            h2obs = copy.deepcopy(obs)
            for j in range(len(h2obs)):
                h2obs[j]['image'] = psuedo_image
            h2a = torch.full((len(obs),), -1)

            h1obs = copy.deepcopy(obss[0])
            h1a = copy.deepcopy(actions[0])
        else:
            h2obs = copy.deepcopy(obss[frame_idx-2])
            h1obs = copy.deepcopy(obss[frame_idx-1])
            h2a = copy.deepcopy(actions[frame_idx-2])
            h1a = copy.deepcopy(actions[frame_idx-1])

            for j in range(len(h1obs)):
                if masks[frame_idx-1][j] == 0:
                    h1obs[j]['image'] = psuedo_image
                    h1a[j] = psuedo_action
                    h2obs[j]['image'] = psuedo_image
                    h2a[j] = psuedo_action
                elif masks[frame_idx-2][j] == 0:
                    h2obs[j]['image'] = psuedo_image
                    h2a[j] = psuedo_action

        preprocessed_obs = preprocess_obss(obs, device=device)
        preprocesses_h1obs = preprocess_obss(h1obs, device = device)
        preprocesses_h2obs = preprocess_obss(h2obs, device = device)

        with torch.no_grad():
            dist, value = model(preprocessed_obs, preprocesses_h2obs, h2a.to(device), preprocesses_h1obs, h1a.to(device))
        
        if argmax:
            action  = torch.argmax(dist.probs, dim = 1)
        else:
            action = dist.sample()

        obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        done = tuple(a | b for a, b in zip(terminated, truncated))

        # Update experiences values

        obss[frame_idx] = obs 

        masks[frame_idx] = mask
        mask = 1 - torch.tensor(done, device=device, dtype=torch.float)
        actions[frame_idx] = action

        h1obss[frame_idx] = h1obs
        h2obss[frame_idx] = h2obs

        h1actions[frame_idx] = h1a
        h2actions[frame_idx] = h2a

        rewards[frame_idx] = torch.tensor(reward, device=device)

         # Update log values

        log_episode_return += torch.tensor(reward, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(procs, device=device)

        for k, done_ in enumerate(done):
            if done_:
                if all_returns[k] is None:
                     all_returns[k] = log_episode_return[k].item()

        log_episode_return *= mask
        log_episode_num_frames *= mask

        frame_idx += 1

        pbar.update(1)

    pbar.close()

    env.close()

    return all_returns


def group_evaluate(model, env_name, all_seeds, argmax = False, max_steps = None):

    n_procs = 5

    num_episodes = len(all_seeds)

    M = num_episodes // n_procs

    all_returns = []

    for i in range(M):
        returns = evaluate(model, env_name, all_seeds[n_procs*i:(n_procs*i+n_procs)], argmax, max_steps)
        all_returns = all_returns + returns

    mean_return = numpy.mean(all_returns)

    print(all_returns)

    success_rate = (sum(numpy.array(all_returns) > 0)/num_episodes) * 100

    model.train()

    return mean_return, success_rate