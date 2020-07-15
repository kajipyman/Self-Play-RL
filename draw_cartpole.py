import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

t = np.arange(0, 1000, 10)
cartpole = np.load("result/cartpole.npy")
cartpole_selfplay = np.load("result/cartpole_selfplay.npy")

ax.set_xlabel('episode')
ax.set_ylabel('reward')
ax.grid()
ax.set_xlim([0, 1000])
ax.set_ylim([0, 210])
ax.plot(t, cartpole, label="cartpole")
ax.plot(t, cartpole_selfplay, label="cartpole (self-play)")
ax.legend(loc=0)
fig.tight_layout()

plt.savefig("result/result_cartpole.png")