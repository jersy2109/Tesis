import os
import gym
import random
import pickle
import datetime
import cv2 as cv
import gym.spaces
import numpy as np
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(env, seed=2109):
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

### Wrappers

class NoopResetEnv(gym.Wrapper):
    """
    Realiza un número random de "NOOP" al invocar reset().
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
    Salta un número de frames y regresa el valor promedio de cada pixel.
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
    Termina el episodio después de un número de pasos.
    Evita que los ambientes entren en un loop o sin moverse.
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
    Realiza la acción "FIRE" para iniciar los juegos que lo requieran.
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
    Reescala las imágenes a 84x84 y las pasa de RGB a gris.
    """
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        num_channels = 1
        
        new_space = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (self._height, self._width, num_channels),
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


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Reescala de 0-255 a 0-1.
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
    Apila los últimos k frames.
    """
    def __init__(self, env, k=4):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.observation_space = gym.spaces.Box(
            low = 0,
            high = 255,
            shape = (4,84,84),
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
        assert len(self.frames) == self.k
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

def make_atari(env_id, frames=4, max_episode_steps=1_000, noop_max=30, skip=4, sample=False):
    """
    Crea el ambiente especificado, pasándolo por los Wrappers especificados.
    """
    env = gym.make(env_id, render_mode=None)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max)
    env = MaxAndSkipEnv(env, skip)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if not sample:
        env = ClipReward(env)
        env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
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

### Agent

class Agent:
    """
     
    """
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()
        
    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
        
    def play_step(self, net, epsilon=0.0, device='cuda'):
        done_reward = None
        
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device) 
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1) # Devuelve el índice de la acción
            action = int(act_v.item())
        
        new_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        
        self.exp_buffer.append(self.state, action, reward, done, new_state)
        self.state = new_state
        
        if done:
            done_reward = self.total_reward
            self._reset()
        
        return done_reward

### Training

def episode_stopping(timer):
    delta = datetime.timedelta(seconds=10)
    if datetime.datetime.now()-timer > delta:
        return True

def training(env_name, replay_memory_size=150_000, max_frames=10_000_000, gamma=0.99, batch_size=32,  \
            learning_rate=0.00025, sync_target_frames=10_000, net_update=4, replay_start_size=50_000, \
            eps_start=1, eps_min=0.1, exp_frames=50_000, seed=2109, device='cuda', verbose=True):
    """
    Función de entrenamiento.
    """
    numberOfDicts = 25

    filename = env_name + "_DQN_" + str(int(exp_frames/1_000)) + 'kFrames_' + str(int(replay_memory_size/1_000)) + "k_" + str(int(max_frames/1_000_000)) + 'M'
    path = "dicts/" + filename
    Path(path).mkdir(parents=True, exist_ok=True)
    
    env = make_atari(env_name + "NoFrameskip-v4")
    buffer = ExperienceReplay(replay_memory_size)
    agent = Agent(env, buffer)
    set_seed(seed=seed, env=env)
    
    net        = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    
    epsilon = eps_start
    eps_decay = (eps_start - eps_min) / exp_frames
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_rewards = []
    loss_history = []
    action_values = []
    tr_finished = True

    best_mean_reward = None
    start_time = datetime.datetime.now()

    for frame in tqdm(range(1, max_frames+1), desc=filename):        
        reward = agent.play_step(net, epsilon, device)
        
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            
            time_passed = datetime.datetime.now() - start_time
            
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), path + "/" + filename + "_Best.dat")
                best_mean_reward = mean_reward

        if len(buffer) < replay_start_size:
            continue

        start_frame = datetime.datetime.now()
        epsilon = max(epsilon-eps_decay, eps_min)

        if frame % net_update == 0:
            sardn = buffer.sample(batch_size)
            batch = Experience(*zip(*sardn))
            
            states_v = torch.tensor(np.array(batch.state)).to(device)
            next_states_v = torch.tensor(np.array(batch.next_state)).to(device)
            actions_v = torch.tensor(batch.action).to(device)
            rewards_v = torch.tensor(batch.reward).to(device)
            done_mask = torch.BoolTensor(batch.done).to(device)
            
            state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            next_state_values = target_net(next_states_v).max(1)[0]
            next_state_values[done_mask] = 0.0
            next_state_values = next_state_values.detach()
            expected_state_action_values = next_state_values*gamma + rewards_v
            
            loss_t = nn.MSELoss()(state_action_values, expected_state_action_values) # MSELoss()(input,target)

            action_values.append((torch.mean(state_action_values)).item())
            loss_history.append(loss_t.item())
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
            
        if frame % sync_target_frames == 0:
            target_net.load_state_dict(net.state_dict())

        if frame % (max_frames // numberOfDicts) == 0:
            if verbose:
                print("{}:  {} games, best result {:.3f}, mean reward {:.3f}, eps {:.2f}, time {}".format(
                    frame, len(total_rewards), max(total_rewards), mean_reward, epsilon, time_passed))
            torch.save(net.state_dict(), path + "/" + filename + '_' + str(int((frame)/(max_frames//numberOfDicts))) + "k.dat")

        if episode_stopping(start_frame):
            print("Taking too long")
            tr_finished = False
            break 

    end_time = datetime.datetime.now() - start_time
    print("Training finished")
    print("{}:  {} games, mean reward {:.3f}, eps {:.2f}, time {}".format(
            frame, len(total_rewards), mean_reward, epsilon, end_time))
         
    pkl_file = "dicts/" + filename + "/" + filename + "_Total.pkl"
    with open(pkl_file, 'wb+') as f:
        pickle.dump(total_rewards, f)
    pkl_file = "dicts/" + filename + "/" + filename + "_Loss.pkl"
    with open(pkl_file, 'wb+') as f:
        pickle.dump(loss_history, f)
    pkl_file = "dicts/" + filename + "/" + filename + "_ActValues.pkl"
    with open(pkl_file, 'wb+') as f:
        pickle.dump(action_values, f)

    parameters = "Environment: {} \
                \nOptical: False \
                \nReplay Memory Size: {} \
                \nMax Frames: {} \
                \nExploration Frames: {} \
                \nGamma: {} \
                \nBatch Size: {} \
                \nLearning Rate: {} \
                \nSync Target Frames: {} \
                \nNet Update: {} \
                \nReplay Start Size: {} \
                \nInitial Epsilon: {} \
                \nMinimum Epsilon: {} \
                \nRandom Seed: {} \
                \nFinished Training: {} \
                \nTraining Time: {}".format(env_name,replay_memory_size,max_frames,exp_frames,gamma,batch_size,learning_rate,
                                        sync_target_frames,net_update,replay_start_size,eps_start,eps_min,seed,tr_finished,end_time)
    
    aux_file = "dicts/" + filename + "/" + filename + "_Parameters.txt"
    with open(aux_file, 'w+') as f:
        f.write(parameters)
   
    return total_rewards, loss_history

# Evaluación

def sample(game, model, model_name, n_samples=30, verbose=True):
    '''
    Obtiene 'n_samples' muestras de la red entrenada.
    '''
    game = game + 'NoFrameskip-v4'
    model = 'dicts/' + model + '/' + model_name
    env = make_atari(game, sample=True, max_episode_steps=None, skip=6, noop_max=1)
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

    if '500kFrames' in directory:
        pkl_file = "samples/" + game + '_DQN_sample_rewards_' + directory.split('_')[-1] + '_T.pkl' #game + "_sample_rewards_1M_T.pkl"
    else:
        pkl_file = "samples/" + game + '_DQN_sample_rewards_' + directory.split('_')[-1] + '.pkl' #game + "_sample_rewards_1M_T.pkl"
        
    with open(pkl_file, 'wb+') as f:
        pickle.dump(game_rewards, f)
    return np.array(game_rewards, dtype=object)


if __name__ == '__main__':
    import sys
    #GAME = sys.argv[1]
    SIZE = 50_000 #int(sys.argv[1])
    EXP_FRAMES = int(sys.argv[1])
    FRAMES = 5_000_000 #int(sys.argv[2])
    Games = ['DoubleDunk', 'Bowling', 'PrivateEye', 'Gravitar', 'Freeway', 'Atlantis', 'Seaquest', 'Pong', 'SpaceInvaders', 'Breakout']
    for game in tqdm(Games):
        path = "dicts/" + game + "_DQN_" +  str(int(EXP_FRAMES/1_000)) + 'kFrames_' + str(int(SIZE/1_000)) + "k_" + str(int(FRAMES/1_000_000)) + 'M'
        if len([x for x in os.listdir(path) if 'dat' in x]) < 26:
            training(env_name=game, replay_memory_size=SIZE, verbose=False, max_frames=FRAMES, exp_frames=EXP_FRAMES)
        sample_model(game=game, directory=path, samples=30)