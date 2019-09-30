"""
Example using this environment with stable_baseline library

from stable_baselines.common.vec_env import DummyVecEnv
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo2_cartpole")

# Enjoy trained agent
model = PPO2.load("ppo2_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
"""

import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.runners import AbstractEnvRunner

from collections import OrderedDict
import numpy as np

from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

scenario_name = "simple_tag"
num_adversaries = 5

class SingleAgentEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
    
    def step(self):
        return self

    def reset(self):
        return self


def env_splitter(multi_env):
    """
    Takes in multiagentenv, and spits out each env individually?
    """
    return [SingleAgentEnv(obs_space, act_space) for obs_space, act_space in zip(multi_env.observation_space, multi_env.action_space)]



class MultiRunner(object):
    def __init__(self, *, env, models, n_steps, gamma, lam):
        """
        A runner to learn the policy of an environment for a model
        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        self.lam = lam
        self.gamma = gamma

        # super().__init__(env=env, model=models, n_steps=n_steps)
        self.env = env
        self.model = model
        n_env = 0 # env.num_envs

        self.batch_ob_shape = []
        self.obs = []
        for idx, env_observation_space in enumerate(env.observation_space):
            self.batch_ob_shape.append((n_env*n_steps,) + env.observation_space.shape)
            self.obs.append(np.zeros((n_env,) + env.observation_space.shape, dtype=env.observation_space.dtype.name))

        obs_reset = env.reset()
        for idx, x in enumerate(obs_reset):
            self.obs[idx][:] = x
        self.n_steps = n_steps
        self.states = [x.initial_state for x in self.model] # get states...
        self.dones = [False for _ in range(n_env)]

    def run(self):
        """
        Run a learning step of the model
        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward


scenario = scenarios.load(scenario_name + ".py").Scenario()
# create world
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
# multi_env = DummyVecMultiEnv([lambda: env]) # TODO and implement??
split_env = env_splitter(env)
agents = [PPO2(MlpPolicy, DummyVecEnv([lambda: x]), verbose=1) for x in split_env]

obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
num_adversaries = env.n # min(env.n, arglist.num_adversaries)
num_adversaries = 0

# get the trainers to train using PPO
