from pathlib import Path
import gym
import gym.spaces
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from tqdm import tqdm
import datetime
import random
import pickle

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
    Apila los k frames observados más recientes.
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


### Ambiente

def make_atari(env_id, frames=4, max_steps=1_000, noop_max=30, skip=4):
    """
    Crea el ambiente con los parámetros especificados.
    """
    env = gym.make(env_id, render_mode=None)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max)
    env = MaxAndSkipEnv(env, skip)
    if max_steps is not None:
        env = TimeLimit(env, max_steps)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipReward(env)
    env = WarpFrame(env)
    env = ScaledFLoatFrame(env)
    env = FrameStack(env, frames)
    env = EpisodicLifeEnv(env)
    return env


### Red

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


### Agente

class Agent:
    """
    El agente que se encarga de jugar.
    """
    def __init__(self, env, exp_buffer, opt=True):
        self.env = env
        self.exp_buffer = exp_buffer
        self.opt = opt
        self._reset()

    def _reset(self):
        if self.opt:
            self.state = self.optical_flow(self.env.reset())
        else:
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
        if self.opt:
            new_state = self.optical_flow(new_state)
        self.total_reward += reward

        self.exp_buffer.append(self.state, action, reward, done, new_state)
        self.state = new_state

        if done:
            done_reward = self.total_reward
            self._reset()
        
        return done_reward
    
    def sample(self, net, directory, file, n_samples=100, verbose=True):
        '''
        Obtiene 'n_samples' número de muestras utilizando la red entrenada.
        '''

        total_reward = []

        for i in range(n_samples):

            self._reset()

            while True:
                reward = self.play_step(net)
                if reward is not None:
                    total_reward.append(reward)
                    break
            
        if verbose:
            print('Game: {}, Average Reward: {}'.format(self.env.unwrapped.spec.id[:-14], np.mean(total_reward)))

        aux_file = directory + "/" + file + "_sampleRewards.txt"
        with open(aux_file, 'a+') as f:
            f.write(str(total_reward) + "\n")


    def optical_flow(self, obs):
        assert np.array(obs).shape == (4, 84, 84)

        first_frame = np.array(obs)[0].astype('uint8')
        #prev_gray = cv.cvtColor(first_frame, cv.COLOR_RGB2GRAY)

        mask = np.zeros((84,84,3))
        mask[..., 1] = 255

        frame = np.array(obs)[-1].astype('uint8')
        #gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        flow = cv.calcOpticalFlowFarneback(first_frame, frame,
                                        flow = None, 
                                        pyr_scale = 0.5, 
                                        levels = 5, 
                                        winsize= 5, 
                                        iterations = 5, 
                                        poly_n = 7, 
                                        poly_sigma = 1.5, 
                                        flags = 0)

        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        flow = cv.cvtColor(mask.astype('float32'), cv.COLOR_HSV2RGB)

        final = np.zeros((4,84,84))
        final[0:3] = mask.reshape(3,84,84)
        final[3] = np.array(obs[1]) / 255.0

        assert final.shape == (4,84,84)

        return final.astype("float32")
    

### Entrenamiento

def episode_stopping(timer):
    delta = datetime.timedelta(seconds=3)
    if datetime.datetime.now()-timer > delta:
        return True
    

def training(env_name, replay_memory_size=50_000, max_frames=5_000_000, gamma=0.99, batch_size=32,  \
            learning_rate=0.00025, sync_target_frames=10_000, net_update=4, replay_start_size=50_000, \
            eps_start=1, eps_min=0.1, seed=2109, device='cuda', verbose=True, opt=True):
    """
    Función de entrenamiento.
    """

    env = make_atari(env_name + "NoFrameskip-v4")
    buffer = ExperienceReplay(replay_memory_size)
    agent = Agent(env, buffer, opt)
    set_seed(seed=seed, env=env)
    
    typeNet = opt*"OPT" + (not opt)*"DQN"
    path = "dicts/" + env_name + "/"
    fileName = env_name + "_" + typeNet + "_" + str(int(replay_memory_size/1_000)) + "k"
    folder = path + "/" + fileName 
    Path(folder).mkdir(parents=True, exist_ok=True)

    net        = DQN((4,84,84), env.action_space.n).to(device)
    target_net = DQN((4,84,84), env.action_space.n).to(device)
    
    epsilon = eps_start
    eps_decay = (eps_start - eps_min) / replay_memory_size
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    tr_finished = True
    total_rewards = []
    loss_history = []

    best_mean_reward = None
    start_time = datetime.datetime.now()

    for frame in tqdm(range(1, max_frames + 1), desc=fileName):

        reward = agent.play_step(net, epsilon, device)

        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            
            time_passed = datetime.datetime.now() - start_time
            
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), path + "/" + fileName + "/" + fileName + "_best.dat")
                best_mean_reward = mean_reward
            
            if len(total_rewards) % 100 == 0:
                agent.sample(net, folder, fileName, n_samples=100, verbose=False)

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
            
            loss_t = nn.HuberLoss()(state_action_values, expected_state_action_values) # MSELoss()(input,target)
            
            loss_history.append(loss_t.item())
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
            
        if frame % sync_target_frames == 0:
            target_net.load_state_dict(net.state_dict())

        if frame % (max_frames // 25) == 0:
            if verbose:
                print("{}:  {} games, best result {:.3f}, mean reward {:.3f}, eps {:.2f}, time {}".format(
                    frame, len(total_rewards), max(total_rewards), mean_reward, epsilon, time_passed))
            torch.save(net.state_dict(), path + "/" + fileName + "/" + fileName + "_" + str(int((frame)/(max_frames//25))) + "k.dat")

        if episode_stopping(start_frame):
            print('Taking too long')
            tr_finished = False
            break
    
    end_time = datetime.datetime.now() - start_time
    print("Training finished")
    print("{}:  {} games, mean reward {:.3f}, eps {:.2f}, time {}".format(
            frame, len(total_rewards), mean_reward, epsilon, end_time))
         
    aux_file = path + "/" + fileName + "/" + fileName + "_total_opt.pkl"
    with open(aux_file, 'wb+') as f:
        pickle.dump(total_rewards, f)
    aux_file = path + "/" + fileName + "/" + fileName + "_loss_opt.pkl"
    with open(aux_file, 'wb+') as f:
        pickle.dump(loss_history, f)

    parameters = "Environment: {} \
                \nOptical: {} \
                \nReplay Memory Size: {} \
                \nMax Frames: {} \
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
                \nTraining Time: {}".format(env_name,opt,replay_memory_size,max_frames,gamma,batch_size,learning_rate,
                                        sync_target_frames,net_update,replay_start_size,eps_start,eps_min,seed,tr_finished,end_time)
    
    aux_file = path + "/" + fileName + "/" + fileName + "_parameters.txt"
    with open(aux_file, 'w+') as f:
        f.write(parameters)

    return total_rewards, loss_history

if __name__ == '__main__':
    import sys
    for game in ["SpaceInvaders", "Pong", "MsPacman"]:
        for size in [50_000, 70_000, 100_000]:
            training(env_name=game, replay_memory_size=size, verbose=False, opt=True)
            training(env_name=game, replay_memory_size=size, verbose=False, opt=False)
        