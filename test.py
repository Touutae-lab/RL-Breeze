import gym
env = gym.make("LunarLander-v2")
for i in range(1000):
    env.step(env.action_space.sample())
    