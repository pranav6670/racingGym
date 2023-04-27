import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import os
gym.logger.set_level(40)

environment_name = "CarRacing-v1"
env = gym.make(environment_name)

model = PPO.load('trained_PPO/PPO_Driving_model')

evaluate_policy(model, env, n_eval_episodes=1, render=True)
env.close()