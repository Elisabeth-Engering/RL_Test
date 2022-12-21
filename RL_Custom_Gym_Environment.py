# Import requirement(s)
import gym
from gym import spaces
import numpy as np
import robosuite as suite

class RoboEnv(gym.Env):
    def __init__(self, 
                 Task = 'Lift', 
                 RenderMode = False):
        super(RoboEnv, self).__init__()
        
        # Initialize environment variables
        self.Task = Task
        self.RenderMode = RenderMode
        
        # Define action and observation space
        self.action_space = spaces.Box(low = -1, # Lower bounds of the intervals
                                       high = 1, # Upper bounds of the intervals
                                       shape = (8,), # Set shape
                                       dtype = np.float32) # Set dtype to float 32
        self.observation_space = spaces.Box(low = -np.inf, # Lower bounds of the intervals
                                            high = np.inf, # Upper bounds of the intervals
                                            shape = (36,), # Set shape
                                            dtype = np.float64) # Set dtype to float 64
         
        # Instantiate the environment
        self.env = suite.make(env_name = self.Task, 
                              robots = "Panda", 
                              has_renderer = self.RenderMode,
                              has_offscreen_renderer = False,
                              horizon = 200, # each episode terminates after 200 steps    
                              use_camera_obs = False) # no observations needed
        
    def step(self, action):
        obs, reward, done, _ = self.env.step(action) # Call the environment step function
        gripper_pos = obs['robot0_eef_pos']
        
        # Create helper functions
        obs = np.hstack((obs['robot0_proprio-state'], self.target_pos)) # Process the observation
        reward = -np.linalg.norm(gripper_pos - self.target_pos[:3]) # Calculate the reward
        #^ Simple reward, negative distance to the target or the inverse of the distance to the target
        
        # done = # Calculate if the episode is done if you want to terminate the episode early
        return obs, reward, done, _
    
    def reset(self):
        obs = self.env.reset() # Call the environment reset function
        
        # Set the target position
        x = np.random.uniform(-0.5, 0.5) 
        y = np.random.uniform(-0.5, 0.5)
        z = np.random.uniform(0.8, 1.3)
        yaw = np.random.uniform(-90, 90)
        self.target_pos = np.array([x, y, z, yaw]) # combine values
        obs = np.hstack((obs['robot0_proprio-state'], self.target_pos)) # Process the observation
        return obs

    def render(self, mode = 'human'):
        self.env.render() # Render the environment to the screen

    def close (self):
        self.env.close() # Close the environment    
        
# Test the environment using SB3's check_env function
from stable_baselines3.common.env_checker import check_env

env = RoboEnv(RenderMode = False) # Instantiate the environment
check_env(env) # Check the environment

print('Environment set up right!')