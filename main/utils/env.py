import gymnasium as gym


def make_env(env_key, seed = None, render_mode = None, max_steps = None):
    if max_steps == None:
        env = gym.make(env_key, render_mode = render_mode)
        env.reset(seed=seed)
        return env
    else:
        env = gym.make(env_key, render_mode = render_mode, max_steps = max_steps)
        env.reset(seed=seed)
        return env
