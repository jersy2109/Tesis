# Imports

import gym
import gym.spaces
import numpy as np
import cv2 as cv
from collections import deque
from tqdm import tqdm
import random
import pickle
import os
from natsort import natsorted
import datetime

import torch
import torch.nn as nn

# Helpers 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed, env):
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# Wrappers

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert self.env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        self.env.reset()
        noops = self.env.unwrapped.np_random.integers(1, self.noop_max+1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset()
        return obs
    

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + self.env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, _, info = self.env.step(action)
            if i == 0: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs            
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs
    

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return obs, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        obs = self.env.reset()
        return obs
    

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = self.env.ale.lives()
        assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(self.env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

    def step(self, action):
        if self.lives > self.env.ale.lives():
            self.lives = self.env.ale.lives()
            action = 1
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
    

class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_r=-1, max_r=1):
        super().__init__(env)
        self.min_r = min_r
        self.max_r = max_r

    def reward(self, reward):
        if reward < 0:
            return self.min_r
        elif reward > 0:
            return self.max_r
        else:
            return 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, self.reward(rew), done, info
    

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        num_channels = 3

        new_space = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (self._height, self._width, num_channels),
            dtype = np.uint8
        )
        original_space = self.observation_space
        self.observation_space = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
       return  cv.resize(
            obs, (self._width, self._height), interpolation=cv.INTER_AREA
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)
    

class FrameStack(gym.Wrapper):
    def __init__(self, env, k=2):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (2,84,84,3),
            dtype = env.observation_space.dtype
        )
        
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.k
        return self.frames
    

class OpticalFlowCV(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = env.observation_space.shape,
            dtype = np.float32
        )

    def observation(self, obs):
        assert np.array(obs).shape == (2, 84, 84, 3)
        frames = obs

        first_frame = np.array(frames)[0].astype('uint8')
        prev_gray = cv.cvtColor(first_frame, cv.COLOR_RGB2GRAY)

        mask = np.zeros_like(first_frame)
        mask[..., 1] = 255

        frame = np.array(frames)[1].astype('uint8')
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                        None, 
                                        0.5, 5, 5, 5, 7, 1.5, 0)

        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        flow = cv.cvtColor(mask, cv.COLOR_HSV2RGB)
        obs[0] = flow
        obs[1] = np.array(frames)[1]
        assert np.array(obs).shape == (2, 84, 84, 3)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def reset(self):
        return self.env.reset()
    

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (6,84,84),
            dtype = np.float32
        )

    def observation(self, obs):
        assert np.array(obs).shape == (2, 84, 84, 3)
        obs = np.array(obs).astype(np.float32)
        obs[1] = obs[1] / 255.0
        obs = obs.reshape(6,84,84)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)
    

class EpisodicLifeEnv(gym.Wrapper):
    """
    Termina el episodio cuando se pierde una vida, pero solo reinicia si
    se pierden todas.
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.is_done = True
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.is_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info
    
    def reset(self):
        if self.is_done:
            obs = self.env.reset()
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs
    
# Ambiente

def make_atari(env_id, max_episode_steps=1_000, noop_max=30, skip=4, sample=False):
    env = gym.make(env_id, render_mode=None)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max)
    env = MaxAndSkipEnv(env, skip)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if sample == False:
        env = ClipReward(env)
        env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = FrameStack(env)
    env = OpticalFlowCV(env)
    env = ScaledFloatFrame(env)
    return env

# RED

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
    
# Evaluación

def sample(game, model, model_name, n_samples=30, verbose=True):
    '''
    Obtiene 'n_samples' muestras de la red entrenada.
    '''
    game = game + 'NoFrameskip-v4'
    model = 'dicts/' + model + '/' + model_name
    env = make_atari(game, sample=True, skip=6)
    net = DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))
    epsilon = 0.05 
    max_time = datetime.timedelta(minutes=5)

    rewards = np.zeros(n_samples)

    for i in range(n_samples):
        game_timer = datetime.datetime.now()
        state = env.reset()
        total_reward = 0.0
        
        while (datetime.datetime.now() - game_timer) < max_time:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_v = torch.tensor(np.array([state], copy=False))
                q_vals = net(state_v).data.numpy()[0]
                action = np.argmax(q_vals)

            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break
        
        if verbose:
            print('Model: {}, Game: {}, Reward: {}'.format(model, i+1,total_reward))

        rewards[i] = total_reward

    return rewards


def get_dats_files(path):
    dats = [x for x in os.listdir(path) if 'dat' in x]
    return natsorted(dats)


def sample_model(game, samples=30, directory=None):
    if directory:
        dats_array = [natsorted([x for x in os.listdir(directory) if 'dat' in x])]
    else:
        dats_array = get_dats_files(path=directory)
    game_rewards = []
    for dats in dats_array:
        model_rewards = []
        mod = '_'.join(dats[0].split('_')[:-1])
        for model in tqdm(dats, desc=mod):
            rw = sample(game=game, model=mod, model_name=model, n_samples=samples, verbose=False)
            model_rewards.append(rw)
        game_rewards.append(model_rewards)

    pkl_file = "samples/" + game + "_RaftSample_rewards.pkl"
    with open(pkl_file, 'wb+') as f:
        pickle.dump(game_rewards, f)
    return np.array(game_rewards, dtype=object)


if __name__ == '__main__':
    import sys
    sample_model(game=sys.argv[1])