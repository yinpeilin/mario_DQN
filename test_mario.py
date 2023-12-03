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

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# monitor_dir = r'./monitor_log/'
# os.makedirs(monitor_dir, exist_ok=True)
# env = Monitor(env, monitor_dir)

env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

save_model_dir = r'result\save_model\best_model\_20000.zip'

model = DQN.load(save_model_dir)

obs = env.reset()
obs = obs.copy()
done = True
while True:
    if done:
        state = env.reset()
    action, _states = model.predict(obs)
    obs, rewards, done, __, info = env.step(action)
    time.sleep(0.01)
    obs = obs.copy()
    env.render()
