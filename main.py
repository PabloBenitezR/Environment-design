from RegattaEnv import RegattaEnv
import time

env = RegattaEnv(distance=200, render_mode="human")
obs, info = env.reset()

done = False
while not done:
    action = 1 if env.time % 10 < 5 else 0
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
    time.sleep(0.05)

env.close()


