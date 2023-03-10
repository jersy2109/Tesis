from pathlib import Path
import gym
import gym.spaces
import numpy as np
import cv2 as cv
from collections import deque, namedtuple
from tqdm import tqdm
import datetime
import random
import pickle
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed, env):
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

### Wrappers

class NoopResetEnv(gym.Wrapper):
    """
    Realiza un número aleatorio de movimientos "NOOP" al reiniciar el ambiente.
    """
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
    """
    Salta un número de frames y regresa el valor máximo de cada pixel.
    """
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._skip = skip
        self._obs_buffer = np.zeros((self._skip,) + self.env.observation_space.shape, dtype=np.uint8)
        
    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, _, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs            
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info
    
    def reset(self):
        return self.env.reset()
    

class TimeLimit(gym.Wrapper):
    """
    Termina el episodio después de un determinado número de pasos.
    Evita que los ambientes se mantengan en un loop o sin jugar. 
    """   
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
        return self.env.reset()


class FireResetEnv(gym.Wrapper):
    """
    Realiza la acción "FIRE" para iniciar los juegos que así lo requieran.
    """
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
        return self.env.step(action)


class ClipReward(gym.RewardWrapper):
    """
    Trunca las recompensas obtenidas a valores entre -1 a 1.
    """
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
    """
    Reescala las imágenes a 84x84 y las convierte a escala de grises.
    """
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        num_colors = 1
        
        new_space = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (self._height, self._width, num_colors),
            dtype = np.uint8,
        )
        original_space = self.observation_space
        self.observation_space = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3
        
    def observation(self, obs):
        frame = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
        return cv.resize(
            frame, (self._width, self._height), interpolation=cv.INTER_AREA
        )
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)
    

class ScaledFLoatFrame(gym.ObservationWrapper):
    """
    Reescala los valores de los pixeles de 0-255 a 0-1.
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 255, 
            shape = self.env.observation_space.shape,
            dtype = np.float32
        )

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)
    

class FrameStack(gym.Wrapper):
    """
    De los más recientes k frames observados, devuelve los últimos 2 apilados.
    """
    def __init__(self, env, k=4):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=self.k)
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (self.k,84,84),
            dtype = self.env.observation_space.dtype,
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
        return self.frames


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
    

### Environment

def make_atari(env_id, frames=4, max_steps=1_000, noop_max=30, skip=4, sample=False, render_mode=None):
    """
    Crea el ambiente con los parámetros especificados.
    """
    env = gym.make(env_id, render_mode=render_mode)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max)
    env = MaxAndSkipEnv(env, skip)
    if max_steps is not None:
        env = TimeLimit(env, max_steps)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if not sample:
        env = ClipReward(env)
        env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = ScaledFLoatFrame(env)
    env = FrameStack(env, frames)
    return env


### Network

class DQN(nn.Module):
    """
    Red Profunda de Aprendizaje Q (Deep Q Network).
    """
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
        o = self.conv(torch.zeros(1,*shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)  
    

### Experience Replay

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'next_state'])

class ExperienceReplay:
    """
    Almacena experiencias pasadas que han sido observadas por el agente.
    Las muestras obtenidas sirven para entrenar la red, buscando minimizar el efecto que tiene la correlación entre pasos.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, *args):
        self.buffer.append(Experience(*args))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    

### Optical Flow

def optical_flow(obs):
    assert np.array(obs).shape == (4, 84, 84)

    first_frame = np.array(obs)[0].astype('uint8')

    mask = np.zeros((84,84,3))
    mask[..., 1] = 255

    frame = np.array(obs)[1].astype('uint8')

    flow = cv.calcOpticalFlowFarneback(first_frame, frame,
                                       flow=None,
                                       pyr_scale=0.5,
                                       levels=5,
                                       winsize=5,
                                       iterations=5,
                                       poly_n=5,
                                       poly_sigma=1.1,
                                       flags=0)
    
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    final = np.zeros((4, 84, 84))
    final[0:3] = mask.reshape(3,84,84)
    final[3] = np.array(obs[1]) / 255.0

    assert final.shape == (4, 84, 84)

    return final.astype('float32')

def sample(env_name, model_folder, n_samples=100, verbose=True):
    '''
    Obtiene 'n_samples' número de muestras utilizando la red entrenada.
    '''
    env_name = env_name + 'NoFrameskip-v4'
    models = [x for x in os.listdir(model_folder) if '.dat' in x]
    rewards = np.zeros((len(models),n_samples))

    for id, model in enumerate(models):
        model_name = model_folder + '/' + model
        env = make_atari(env_name, sample=True)
        net = DQN((4,84,84), env.action_space.n)
        net.load_state_dict(torch.load(model_name, map_location=lambda storage, loc:storage))

        for i in tqdm(range(n_samples), desc=model):

            state = env.reset()
            total_reward = 0.0

            while True:
                
                if 'OPT' in model:
                    state = optical_flow(state)
                state_v = torch.tensor(np.array([state], copy=False))
                q_vals = net(state_v).data.numpy()[0]
                action = np.argmax(q_vals)

                state, reward, done, _ = env.step(action)
                total_reward += reward
                if reward > 0:
                    print(reward)
                if done:
                    break
            
            rewards[id, i] = total_reward

        if verbose:
            print('Game: {}, Average Reward: {}'.format(model, np.mean(rewards[id])))

    file_name = 'Sampling_Results_' + model_folder



if __name__ == '__main__':
    sample(env_name=sys.argv[1], model_folder=sys.argv[2], n_samples=20)