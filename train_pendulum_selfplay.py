import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np


env_self = gym.make('Pendulum-v0')
env_opp = gym.make('Pendulum-v0')

obs_space = env_self.observation_space
obs_size = obs_space.low.size
action_space = env_self.action_space
action_size = action_space.low.size
q_func = chainerrl.q_functions.FCQuadraticStateQFunction(
            obs_size, action_size,
            n_hidden_channels=100,
            n_hidden_layers=2,
            action_space=action_space)

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

# Set the discount factor that discounts future rewards.
gamma = 1

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env_self.action_space.sample)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100, phi=phi)

n_episodes = 1000
max_episode_len = 200
Rs = []
for i in range(1, n_episodes + 1):
    obs_self = env_self.reset()
    obs_opp = env_opp.reset()
    reward = 0
    fake_reward_self = 0
    fake_reward_opp = 0
    done = False
    done_self = False
    done_opp = False
    fake_R_self = 0  # return (sum of rewards)
    fake_R_opp = 0  # return (sum of rewards)
    t = 0  # time step
    while not done and t < max_episode_len:

        if not done_self:
            action_self = agent.act_and_train(obs_self, reward)
            obs_self, fake_reward_self, done_self, _ = env_self.step(action_self)
            fake_R_self += fake_reward_self
        
        if not done_opp:
            action_opp = agent.act(obs_opp)
            obs_opp, fake_reward_opp, done_opp, _ = env_opp.step(action_opp)
            fake_R_opp += fake_reward_opp
        
        if done_self and done_opp:
            done = True
            if fake_R_self > fake_R_opp:
                reward = 1
            elif fake_R_self == fake_R_opp:
                reward = 0
            else:
                reward = -1

        t += 1

    if i % 10 == 0:
        Rs.append(fake_R_self)
        print('episode:', i,
              'R:', fake_R_self,
              'statistics:', agent.get_statistics())
    agent.stop_episode_and_train(obs_self, reward, done)

np.save("result/pendulum_selfplay.npy", Rs)
print('Finished.')