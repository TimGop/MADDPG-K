import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
eps = np.array([i*100 for i in range(1,501)])
tot_rew = []
for i in range(1,5):
    with open(f"src/MADDPG/DDPGpaper{i}_rewards.pkl", 'rb') as f:
        rewards = np.array(pickle.load(f))
        tot_rew.append(rewards)
rew_m = np.stack(tot_rew).mean(axis = 0)
rew_std = np.stack(tot_rew).std(axis = 0)
plt.plot(eps, rew_m, color='black', label="policy network", linewidth = 0.6)
plt.fill_between(eps, rew_m-rew_std, rew_m + rew_std,facecolor='lightskyblue',alpha=0.7, edgecolor = 'deepskyblue', linewidth=0.5)
plt.xlabel('# episodes')
plt.ylabel('mean episode reward')
plt.title('Mean of sum of rewards after each update:')
plt.legend()
plt.savefig(f"paper_MADDPG_rewards_mean.png")