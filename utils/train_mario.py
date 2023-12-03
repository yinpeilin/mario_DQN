
from sys_path import *

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY,COMPLEX_MOVEMENT
import time
from matplotlib import pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor, DummyVecEnv
from stable_baselines3 import DQN

from gymnasium.wrappers import GrayScaleObservation

from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from model.mario_arch import CustomCNN
import cv2
from gymnasium import spaces
import numpy as np
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self.observation_space = spaces.Box(low=0,high=255,shape=(128,120,3),dtype = np.uint8)
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
            
        obs = cv2.resize(obs, (obs.shape[0]//2,obs.shape[1]//2))
        # return obs, total_reward/100, done, False, info
        return obs, total_reward/100, done, False, info
    def reset(self,seed = None, options = None):
        obs,__ = self.env.reset()
        obs = cv2.resize(obs, (obs.shape[0]//2, obs.shape[1]//2))
        return obs,{'test':'123'}
        

def make_env():
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    
    env = SkipFrame(env, 4)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    # use a seed for reproducibility
    # Important: use a different seed for each environment
    # otherwise they would generate the same experiences
    env.reset()
    return env
    
    
if __name__ == '__main__':
    worker_num = 50

    log_dir = './result/log/'
    os.makedirs(log_dir, exist_ok=True)

    env = SubprocVecEnv([make_env for i in range(worker_num)])
    # env = DummyVecEnv([make_env])
    env = VecMonitor(env, log_dir)
    env = VecFrameStack(env, 4, channels_order='last')

    class SaveOnStepCallback(BaseCallback):
        def __init__(self, check_freq, save_path, verbose=1):
            super(SaveOnStepCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.save_path = os.path.join(save_path, 'best_model')

        def _init_callback(self):
            # Create folder if needed
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self):
            if self.n_calls % self.check_freq == 0:
                self.model.save(os.path.join(
                    self.save_path, '_{}'.format(self.n_calls)))

            return True


    learning_rate = 1e-4

    tensorboard_log = r'./result/log'
    
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=2048),
        net_arch=[512,256],
    )
    model = DQN("CnnPolicy", env, verbose=1,learning_starts = 10000,
                learning_rate=learning_rate,batch_size=4096, buffer_size = 100000,
                tau = 0.99,gamma = 0.8,target_update_interval = 10000,
                tensorboard_log=tensorboard_log,
                exploration_fraction = 0.2,exploration_final_eps = 0.05, exploration_initial_eps = 0.6,
                policy_kwargs=policy_kwargs)
    # model.load(r"result\save_model\best_model\_60000.zip")
    save_path = r"./result/save_model"
    callback1 = SaveOnStepCallback(check_freq=20000, save_path=save_path)
    model.learn(total_timesteps=10000000, callback=callback1,progress_bar=True)
