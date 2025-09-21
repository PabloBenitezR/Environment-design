from RegattaRLEnv import RegattaRLEnv, RegattaConfig
import numpy as np

env = RegattaRLEnv(RegattaConfig(distance_total=500.0, reward_mode="vector"))
obs, info = env.reset()
done = False
R = np.zeros(2, dtype=np.float64)
while not done:
    action = env.action_space.sample()  # replace with your policy
    obs, r, term, trunc, info = env.step(action)
    R += r  # r is a 2D vector
    done = term or trunc
print("Episode returns (time, energy):", R)
