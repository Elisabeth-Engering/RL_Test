# Import requirement(s)
from stable_baselines3 import PPO
import gym
import time
from RL_Custom_Gym_Environment import RoboEnv # import custom env

from clearml import Task

task = Task.init(project_name='2022-Y2B-RoboSuite/Elisabeth', task_name = 'Experiment1') # Set project name
task.set_base_docker('deanis/robosuite:py3.8-2') # Setting the base docker image
task.execute_remotely(queue_name = 'default') # Setting the task to run remotely on the default queue

# Define the arguments:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type = float, default = 0.0003)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--n_steps', type = int, default = 2048)
parser.add_argument('--n_epochs', type = int, default = 10)
args = parser.parse_args()

# Set API key using os environment variables
import os
os.environ['WANDB_API_KEY'] = 'fa33a4fcc33d94c6d5d4221796ed2942a0bc6117'

env = RoboEnv() 
obs = env.reset()[0]

import wandb
from wandb.integration.sb3 import WandbCallback
run = wandb.init(project = 'custom_control_test', sync_tensorboard = True) # Initialize wandb project

# Add tensorboard logging to the model
model = PPO('MlpPolicy', 
            env, 
            verbose = 1, 
            learning_rate = args.learning_rate, 
            batch_size = args.batch_size,
            n_steps = args.n_steps,
            n_epochs = args.n_epochs, 
            tensorboard_log = f'runs/{run.id}')

# Create wandb callback
wandb_callback = WandbCallback(model_save_freq = 1000,
                                model_save_path = f'models/{run.id}',
                                verbose = 2)

# Variable for how often to save the model
timesteps = 100000
for i in range(10):
    # Add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # Add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps = timesteps, 
                callback = wandb_callback, # Add wandb callback to the model training
                progress_bar = True, 
                reset_num_timesteps = False,
                tb_log_name = f"runs/{run.id}")
    
    # Save the model to the models folder with the run id and the current timestep
    model.save(f"models/{run.id}/{timesteps*(i+1)}")