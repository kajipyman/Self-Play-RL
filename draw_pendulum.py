import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

t = np.arange(0, 1000, 10)
pendulum1 = np.load("result/pendulum1.npy")
pendulum_selfplay1 = np.load("result/pendulum_selfplay1.npy")
pendulum2 = np.load("result/pendulum2.npy")
pendulum_selfplay2 = np.load("result/pendulum_selfplay2.npy")

ax.set_xlabel('episode')
ax.set_ylabel('reward')
ax.grid()
ax.set_xlim([0, 1000])
ax.set_ylim([-2000, 0])
ax.plot(t, pendulum1, label="pendulum")
ax.plot(t, pendulum_selfplay1, label="pendulum (self-play)")
ax.plot(t, pendulum2, label="pendulum")
ax.plot(t, pendulum_selfplay2, label="pendulum (self-play)")
ax.legend(loc=0)
fig.tight_layout()

plt.savefig("result/result_pendulum.png")