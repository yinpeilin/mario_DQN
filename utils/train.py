import gymnasium as gym
import numpy as np
from env.GameEnv import CustomEnv 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
if __name__ == "__main__":
    vec_env = make_vec_env(CustomEnv,n_envs=10,
                           env_kwargs={"PlaneGame":PlaneGame, "json_data_path":"assets/plane_game.json", "render_mode":"rgb_array"})
    
        
        # if i!=0:
    model = PPO.load("best_model",vec_env)
    # print(model.learning_rate)
        # else:
    # policy_kwargs = dict(features_extractor_class = CustomCombinedExtractor,net_arch=dict(pi=[1000, 200], vf=[1000, 200]))
    # policy_kwargs = dict(features_extractor_class = CustomCombinedExtractor)
    # policy_kwargs = None
    # model = PPO("MultiInputPolicy", vec_env, verbose=1,learning_rate=0.0001,policy_kwargs= policy_kwargs )
    print(model.policy)
    # print(model)
    env = CustomEnv(PlaneGame, "assets/plane_game.json","rgb_array")
    eval_callback = EvalCallback(env, best_model_save_path='./', log_path='./eval_log/',eval_freq=5000,render=False, deterministic=True,n_eval_episodes=10)
    
    tmp_path = "./train_log/"
    new_logger = configure(tmp_path, ["log", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=10000000,progress_bar=True,callback= eval_callback)
    # model.save("PPO_plane_game"+str(i))

    del model # remove to demonstrate saving and loading

    env = CustomEnv(PlaneGame, "assets/plane_game.json")
    model = PPO.load("best_model")
    obs,_ = env.reset()
    while True:
        action, _states = model.predict(obs)
        print(action)
        obs, rewards, dones, _,info = env.step(action)
        env.render()
        
        if dones == True:
            obs,_ = env.reset()
