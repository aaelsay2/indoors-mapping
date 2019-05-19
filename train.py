from MapNav import MapNav
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import cv2

from replay_memory import ReplayMemory
from DQN import DQN

import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.nn.utils import clip_grad_value_

from tensorboardX import SummaryWriter

import logging

logging.basicConfig(level='INFO',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

BATCH_SIZE = 128
GAMMA = 0.999
GRAD_CLIP = 1
TARGET_UPDATE = 100
NUM_FRAMES = 50000000
LEARNING_RATE = 1e-4

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = './dqn-indoor_map.model'

transform = T.Compose([T.ToPILImage(),
                    T.Grayscale(),
                    T.ToTensor()])

plt.ion()

env = MapNav(map_id=1, safe_offset=3, sensor_range=20, fov=60, resolution=50)

# Get number of actions from gym action space
n_actions = len(env.get_possible_actions())

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = env.render(render_scale=1, complete=False, store=False)
H, W = init_screen.shape

policy_net = DQN(H, W, n_actions).to(device)
target_net = DQN(H, W, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


writer = SummaryWriter()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

optimizer = optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE)
# optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

MEMORY_CAPACITY = 10000
memory = ReplayMemory(MEMORY_CAPACITY)


def get_screen():
  screen = env.render(render_scale=1, complete=False, store=False)
  H,W = screen.shape
  # screen = np.reshape(screen,(1, H, W))
  # print(screen.shape)
  screen = np.ascontiguousarray(screen, dtype=np.float32)
  screen = torch.from_numpy(screen)
  screen = transform(screen.unsqueeze(0))
  # pdb.set_trace()
  return screen.to(device)
  

def preprocess_frame(frame):
    # frame = torch.from_numpy(frame)
    # frame = frame.to(device, dtype=torch.float32)
    frame = frame.unsqueeze(0)

    return frame


def plot_game():
    img = env.render()
    # img = get_screen().squeeze(0).cpu().numpy()
    cv2.imshow('map',img)
    cv2.waitKey(20)


#   plt.figure(1)
#   plt.clf()
#   # pdb.set_trace()

#   plt.imshow(img, interpolation='none',cmap='gray')
#   plt.show()
#   plt.pause(0.0001)  # pause a bit so that plots are updated




steps_done = 0

episode_durations = []

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.tensor(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    # pdb.set_trace()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


if __name__=='__main__':
  num_episodes = 1000000
  recording_limit = 2000
  recording_count = 0
  env.start_recording()

  state = get_screen()

  logging.info('Filling up memory')
  for t in range(MEMORY_CAPACITY):
    # sample action from observed state
    action = torch.LongTensor(1).random_(n_actions).to(device).view(1, 1) #torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    reward, collision, done = env.step(action.item()) #

    next_state = get_screen()
    if(done or collision):
        next_state = None

    memory.push(state, action, next_state, reward)
    state = next_state

    if(done or collision):
        env.reset()
        state = get_screen()

    if((t % 50000) == 0):
        logging.info('finished {:.02f} %'.format(t / MEMORY_CAPACITY * 100))

  logging.info('Training Start')

  total_frame_count = 0

  for i_episode in range(num_episodes):
  # Initialize the environment and state
    env.reset()
    current_screen = get_screen()
    state = current_screen
    policy_net.train()

    total_reward = 0
    episode_update_reward = 0
    episode_update = 0    

    for t in count():
    # Select and perform an action
      action = policy_net.get_greedy_action(state)
      reward, collision, done = env.step(action.item()) #.item()
      total_reward += reward

      reward = torch.tensor([reward], device=device)

      current_screen = get_screen()
      next_state = current_screen
      if done or collision:        
        next_state = None

      # plt.figure(1)
      # plt.imshow(current_screen.cpu().squeeze(0).squeeze(0).numpy(),
          #  interpolation='none')

      # Store the transition in memory
      memory.push(state, action, next_state, reward)

      # Move to the next state
      state = next_state

      # Perform one step of the optimization (on the target network)
      loss = optimize_model()

      plot_game()
      # if recording_count<=recording_limit:
      #   env.render(render_scale=10, complete=True, store=True)
      #   recording_count=recording_count+1
      #   if recording_count==recording_limit:
      #     print('sample video recorded')
      #     env.end_recording()

      # update episode graph variables
      episode_update_reward += reward.item()
      episode_update += 1
      total_frame_count += 1
      writer.add_scalar('data/loss', loss.item(), total_frame_count)
      writer.add_scalar('data/eps', policy_net._get_eps(), total_frame_count)

      # Update the target network, copying all weights and biases in DQN
      if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save(policy_net, MODEL_PATH)

      if done or collision:
        episode_durations.append(t + 1)
        plot_durations()
        break

    episode_update_reward /= episode_update
    writer.add_scalar('data/episode_update_reward', episode_update_reward, i_episode)
    writer.add_scalar('data/episode_reward', total_reward, i_episode)
    writer.add_scalar('data/episode_length', episode_update, i_episode)

    # create video every 100 episodes
    # if((i_episode % 100) == 0):
    #     policy_net.eval()
    #     env.reset()
    #     current_screen = get_screen()
    #     state = current_screen

    #     episode_video_frames = []
    #     for t in count():
    #         action = policy_net.get_greedy_action(state, update_step=False)
    #         _, _, done = env.step(action.item()) #.item()
    #         obs = get_screen()
    #         episode_video_frames.append(obs.cpu().numpy())
    #         if(done or t > 3000):
    #             break
    #     # stacked with T, C, H, W     #T, H, W, C
    #     # pdb.set_trace()
    #     stacked_frames = np.stack(episode_video_frames).transpose(1, 0, 2, 3)
    #     stacked_frames = np.expand_dims(stacked_frames, 0)
    #     # video takes B, C, T, H, W
    #     writer.add_video('video/episode', stacked_frames, i_episode)

    if(total_frame_count > NUM_FRAMES):
        torch.save(policy_net, MODEL_PATH)
        break      

    print('Reward this episode:',total_reward)


  print('Complete')
