import numpy as np
import random
from collections import namedtuple, deque

from model import DuelingDQN

import torch
import torch.nn.functional as F
import torch.optim as optim





from utils.data_structures import SegmentTree, MinSegmentTree, SumSegmentTree

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DuelingDQN(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingDQN(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        self.priority_alpha = 0.0#current best: 03
        self.priority_beta_start = 0.4
        self.priority_beta_frames = BUFFER_SIZE

        # Replay memory
        self.memory = PrioritizedReplayMemory(BUFFER_SIZE, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.push((state, action, reward, next_state, done))
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.storage_size() > BATCH_SIZE:
                #print("storage == ", self.memory.storage_size())
                experiences, idxes, weights = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, idxes, weights, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, idxes, weights, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = torch.from_numpy(np.vstack([state for state in states if state is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([action for action in actions if action is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([reward for reward in rewards if reward is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([next_state for next_state in next_states if next_state is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([done for done in dones if done is not None]).astype(np.uint8)).float().to(device)

        # Get max predicted Q values (for next states) from target model
        #print("state-action values:")
        #print(self.qnetwork_target(next_states).detach())
        #print(next_states)
        next_target_Q = self.qnetwork_target.forward(next_states)
        #print("next_target_Q == ", next_target_Q)
        
        _,next_local_Q_index = torch.max(self.qnetwork_local.forward(next_states),axis = 1)
        
        #Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        Q_targets_next = next_target_Q[range(next_target_Q.shape[0]),next_local_Q_index]
        
        Q_targets_next1 = Q_targets_next.reshape((len(Q_targets_next),1))
        
        
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next1 * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        #print(Q_expected)
        #print(Q_targets)
        
        diff = Q_expected - Q_targets
        #print(diff)
        #diff = diff.mean()
        #print(idxes)
        #print(diff.detach().squeeze().abs().cpu().numpy().tolist())
        #update the priority of the replay buffer
        
        
        self.memory.update_priorities(idxes,diff.detach().squeeze().abs().cpu().numpy().tolist())

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) * weights
        loss = loss.mean()
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class PrioritizedReplayMemory(object):
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        super(PrioritizedReplayMemory, self).__init__()
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        assert alpha >= 0
        self._alpha = alpha

        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame=1

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, data):
        idx = self._next_idx

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize


        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _encode_sample(self, idxes):
        return [self._storage[i] for i in idxes]

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        idxes = self._sample_proportional(batch_size)

        weights = []

        #find smallest sampling prob: p_min = smallest priority^alpha / sum of priorities^alpha
        p_min = self._it_min.min() / self._it_sum.sum()

        beta = self.beta_by_frame(self.frame)
        self.frame+=1
        
        #max_weight given to smallest prob
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = torch.tensor(weights, device=device, dtype=torch.float) 
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample, idxes, weights

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = (priority+1e-5) ** self._alpha
            self._it_min[idx] = (priority+1e-5) ** self._alpha

            self._max_priority = max(self._max_priority, (priority+1e-5))
            
    def storage_size(self):
        return len(self._storage)