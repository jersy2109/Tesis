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
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed, env):
    env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

### Wrappers

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
    

### Environment

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
    env = WarpFrame(env)
    env = FrameStack(env)
    env = OpticalFlowCV(env)
    env = ScaledFloatFrame(env)
    env = EpisodicLifeEnv(env)
    return env


### Network 

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


### Experience Replay

Experience = namedtuple('Experience', field_names=['state','action','reward','done','next_state'])

class ExperienceReplay:
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
    def __init__(self, env, exp_buffer, val=False):
        self.env = env
        self.exp_buffer = exp_buffer
        self.val = val
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
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        
        if self.val == False:
            self.exp_buffer.append(self.state, action, reward, done, new_state)
        
        self.state = new_state

        if done:
            done_reward = self.total_reward
            self._reset()

        return done_reward


### Training

def training(env_name, replay_memory_size=75_000, max_frames=50_000_000, gamma=0.99, batch_size=32,  \
            learning_rate=0.00025, sync_target_frames=10_000, net_update=4, replay_start_size=50_000, \
            eps_start=1, eps_min=0.1, seed=2109, device='cuda', verbose=True):
    """
    Función de entrenamiento.
    """
    path = "dictsOpt/" + env_name + "_opt"
    Path(path).mkdir(parents=True, exist_ok=True)
    
    env = make_atari(env_name, max_episode_steps=5_000)
    buffer = ExperienceReplay(replay_memory_size)
    agent = Agent(env, buffer)
    set_seed(seed=seed, env=env)
    
    net        = DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment='-' + env_name + "_opt")
    
    epsilon = eps_start
    eps_decay = (eps_start - eps_min) / replay_memory_size
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    total_rewards = []
    val_rewards = []
    loss_history = []
    loss_t = None

    best_val_reward = None
    start_time = datetime.datetime.now()

    for frame in tqdm(range(1, max_frames+1), desc=env_name):        
        reward = agent.play_step(net, epsilon, device)

        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            
            time_passed = datetime.datetime.now() - start_time
            
            #writer.add_scalar("epsilon", epsilon, frame)
            writer.add_scalar("reward_100", mean_reward, frame)
            writer.add_scalar("reward", reward, frame)
           
            if len(total_rewards) % 1000 == 0:
                test_env = make_atari(env_name, sample=True)
                test_agent = Agent(test_env, buffer, True)
                test_dones = 0
                tot_val_rew = 0
                while test_dones < 50:
                    rw = test_agent.play_step(net, 0.05, device)
                    if rw is not None:
                        tot_val_rew += rw
                        test_dones += 1
#                        print("Test reward {}".format(rw))
                mean_val_rw = tot_val_rew / 50
                test_env.close()
                if best_val_reward is None or best_val_reward < mean_val_rw:
                    torch.save(net.state_dict(), path + "/" + env_name + "_opt_best.dat")
                    best_val_reward = mean_val_rw
                
                val_rewards.append(mean_val_rw)
#                print("Average reward in 50 games: {:.2f}".format(mean_val_rw))
#                
        if len(buffer) < replay_start_size:
            continue

        epsilon = max(epsilon-eps_decay, eps_min)

        if (frame+1) % net_update == 0:
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

            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()
        
        if (frame + 1) % sync_target_frames == 0:
            target_net.load_state_dict(net.state_dict())
            if loss_t is not None:
                loss_history.append(loss_t)

        if (frame) % (max_frames / 10) == 0:
            if verbose:
                print("{}:  {} games, best result {:.3f}, mean reward {:.3f}, eps {:.2f}, time {}".format(
                    frame + 1, len(total_rewards), max(total_rewards), mean_reward, epsilon, time_passed))

            torch.save(net.state_dict(), path + "/" + env_name + "_opt_" + str(int((frame+1)/(5_000_000))) + ".dat")

    print("Training finished")
    print("{}:  {} games, mean reward {:.3f}, eps {:.2f}, time {}".format(
            frame + 1, len(total_rewards), mean_reward, epsilon, time_passed))
         
    writer.close()
    pkl_file = "dicts/" + env_name + "/" + env_name + "_loss_opt.pkl"
    with open(pkl_file, 'wb+') as f:
        pickle.dump(loss_history, f)
    pkl_file = "dicts/" + env_name + "/" + env_name + "_total_opt.pkl"
    with open(pkl_file, 'wb+') as f:
        pickle.dump(total_rewards, f)
    pkl_file = "dicts/" + env_name + "/" + env_name + "_val_opt.pkl"
    with open(pkl_file, 'wb+') as f:
        pickle.dump(val_rewards, f)
    return total_rewards, val_rewards, loss_history


if __name__ == '__main__':
    import sys
    training(env_name=sys.argv[1]+'NoFrameskip-v4', verbose=False)