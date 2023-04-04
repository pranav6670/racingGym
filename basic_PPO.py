import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


env = gym.make('CartPole-v1')


model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model# remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

