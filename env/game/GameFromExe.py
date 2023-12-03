import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


class Game():
    pass



class CustomEnv(gym.Env):

    def __init__(self, PlaneGame, json_data_path, render_mode="human"):
        super().__init__()

        self.render_mode = render_mode
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(5)
        # Example for using image as input (channel-first; channel-last also works):
        self.input_num = 1

        self.observation_space = spaces.Dict(
            {
                "direction": spaces.Box(-1, 1, shape=(2*200,), dtype=np.float32),
                "distance": spaces.Box(0, 1, shape=(200,), dtype=np.float32),
                "info":spaces.Box(0, 1, shape=(200,), dtype=np.float32),
                "one_dimension_data": spaces.Box(-1, 1, shape=(6,), dtype=np.float32),
            }
        )

        self.Game_class = PlaneGame
        self.json_data_path = json_data_path
        self.game = None
        self.game_window = None

        self.frame = np.zeros((1000, 1000,3), dtype=np.uint8)
        self.state = None
        self.one_dimension_data = np.zeros((6), dtype=np.float32)
        self.direction = np.zeros(
            (2*200), dtype=np.float32)
        self.distance_info = np.zeros(
            (2*200), dtype=np.float32)

        self.info = {"reward": 0.0}

    def _get_obs(self):
        
        # 能观测到的信息
        one_dimension_data,distance,info,direction, __, __ = self.game.GameStateGet()
        direction = np.reshape(direction,(400,))
        distance = np.reshape(distance, (200,))
        info = np.reshape(info, (200,))
        return {"distance": distance,"info": info,"direction":direction, "one_dimension_data": one_dimension_data}

    def _get_info(self):
        
        # 奖励值
        return {
            "reward": self.game.reward
        }

    def step(self, action):

        self.game.ActionSet(action)
        self.game.StateStep()
        
        
        
        return self._get_obs(), self.game.reward, self.game.GameOver(), False, self._get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game = self.Game_class(self.json_data_path)

        self.game.GameStart()

        return self._get_obs(), self._get_info()

    def render(self):

        if self.render_mode == "human":
            self._render_frame()
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        
        if self.game_window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.game_window = pygame.display.set_mode(
            (1000, 1000)
                )
        
        
        if self.render_mode == "human":
            
            canvas = pygame.Surface((1000, 1000))
            canvas.fill((0, 0, 0))
        
        
        
            self.game.FrameShow(canvas)
        # The following line copies our drawings from `canvas` to the visible window
            self.game_window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.frame = pygame.surfarray.pixels3d(canvas)
            
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
        else:  # rgb_array
            # self.frame = pygame.surfarray.pixels3d(canvas)
            
            return self.frame

    def close(self):
        if self.game_window is not None:
            pygame.display.quit()
            pygame.quit()
