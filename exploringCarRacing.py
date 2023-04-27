import gym
import numpy as np
# Define the Environments
env = gym.make('CarRacing-v0').env

# Number of Dimensions in the Observable Space and number of Control Actions in the Environments
print('Observation Space:', env.observation_space)
print('Action Space:', env.action_space)

print("\n")
print("Observation Space Param: 96x96x3 values for Red, Green and Blue pixels")
print("Observation Space Highs:", np.mean(env.observation_space.high))
print("Observation Space Lows:   ", np.mean(env.observation_space.low))