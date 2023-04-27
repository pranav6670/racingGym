import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os

environment_name = "CartPole-v0"
env = gym.make(environment_name)

save_path = os.path.join('Training', 'Saved Models')
log_path = os.path.join('Training', 'Logs')
training_log_path = os.path.join(log_path, 'PPO_3')

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=190, verbose=1)
eval_callback = EvalCallback(env, 
                             callback_on_new_best=stop_callback, 
                             eval_freq=10000, 
                             best_model_save_path=save_path, 
                             verbose=1)

model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)

model.learn(total_timesteps=20000, callback=eval_callback)

model_path = os.path.join('Training', 'Saved Models', 'best_model')
model = PPO.load(model_path, env=env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()