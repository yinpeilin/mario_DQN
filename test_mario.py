'''
this is not still not work.
'''

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
from matplotlib import pyplot as plt
from gymnasium.wrappers import GrayScaleObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
import os
from stable_baselines3 import DQN
import time
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import cv2
import gymnasium as gym
from gymnasium import spaces
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(128, 120, 3), dtype=np.uint8)
        self._skip = skip
        # self.time = 0

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, __, info = self.env.step(action)
            total_reward += reward
            if done:
                # self.time+=1
                self.env.reset()
                # if self.time % 10 == 0:
                #     done = True
                # else:
                #     done = False
                break

        obs = cv2.resize(obs, (obs.shape[0]//2, obs.shape[1]//2))
        # return obs, total_reward/100, done, False, info
        return obs, total_reward/100, done, False, info

    def reset(self, seed=None, options=None):
        obs, __ = self.env.reset()
        obs = cv2.resize(obs, (obs.shape[0]//2, obs.shape[1]//2))
        return obs, {'test': '123'}


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, 4)
# monitor_dir = r'./monitor_log/'
# os.makedirs(monitor_dir, exist_ok=True)
# env = Monitor(env, monitor_dir)

env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

save_model_dir = r'result\save_model\best_model\_140000.zip'

model = DQN.load(save_model_dir)

obs = env.reset()
obs = obs.copy()
done = True
while True:
    if done:
        state = env.reset()
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print(obs.shape)
    cv2.imshow('obs',obs[0,:,:,-1])
    cv2.waitKey(10)
    # time.sleep(0.01)
    # obs = obs.copy()
    # env.render()
