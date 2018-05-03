import numpy as np
from agents.replaybuffer import ReplayBuffer
from agents.actor import Actor
from agents.critic import Critic
import tensorflow as tf

class Agent():
    def __init__(self, cfg):
        # Environment configuration
        self.action_shape = cfg['env']['action_shape']

        # Replay memory
        cfg['agent']['memory']['action_shape'] = self.action_shape
        self.memory = ReplayBuffer(**cfg['agent']['memory'])

        # Algorithm parameters
        self.exploration_mu, self.exploration_sigma = cfg['agent']['noise']


        self.gamma = cfg['agent']['gamma']
        self.tau = cfg['agent']['tau']

        state_flatten_shape = [np.prod(self.memory.flatten_state_shape)]
        # Actor Model
        self.actor = Actor(state_flatten_shape, self.action_shape, cfg['env']['action_range'],
                           self.tau, self.memory.batch_size, cfg['actor'])

        # Critic Model
        self.critic = Critic(state_flatten_shape, self.action_shape, self.tau, cfg['critic'])

        # Flag & Counter
        self.training = True
        self.episode = 0
        self.max_episode_explore = cfg['agent']['explore']

    def init_actor_critic(self):
        # Initialize target model
        self.critic.copy_local_in_target()
        self.actor.copy_local_in_target()

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward,
                        next_state, done, self.training)

    def act(self, state):
        self.last_state = state

        window_states = state.reshape(1, -1)
        action = self.actor.predict(window_states)

        if self.training and self.episode < self.max_episode_explore:
            p = self.episode / self.max_episode_explore
            action = p * action + (p-1) * np.random.normal(self.exploration_mu, self.exploration_sigma)
        return np.clip(action.ravel(), a_max=900, a_min=0)

    def learn(self):
        if self.memory.is_sufficient():
            experiences = self.memory.sample()

            states = experiences['state'][:, 0].reshape(self.memory.batch_size, -1)
            actions = experiences['action'][:, 0].reshape(self.memory.batch_size, -1)
            rewards = experiences['reward']
            dones = experiences['done']
            next_states = experiences['next_state'][:, 0].reshape(self.memory.batch_size, -1)

            # get predicted next state action and Q values from target models
            actions_next = self.actor.get_targets(next_states)
            Q_targets_next = self.critic.get_targets(next_states, actions_next)

            # Compute Q targets for current states and train critic model
            Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
            self.critic.fit(states, actions, Q_targets)

            # Train actor model
            action_gradients = self.critic.get_actions_grad(states, actions)[0]
            self.actor.fit(states, action_gradients)

            # Soft-update target models
            self.critic.soft_update()
            self.actor.soft_update()
