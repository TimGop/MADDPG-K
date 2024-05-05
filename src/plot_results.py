import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv_path = "results.csv"


def plot_results():
    df = pd.read_csv(csv_path, sep=",", header=None)
    per_agent_rewards = df.iloc[1].to_numpy()
    per_agent_rewards = [np.fromstring(per_agent_rewards[i][2:-1], dtype=np.float32, sep=" ") for i in
                         range(per_agent_rewards.shape[0])]
    for i, a in enumerate(per_agent_rewards):
        if a.shape[0] != 4:
            per_agent_rewards[i] = per_agent_rewards[i - 1]
    per_agent_rewards = np.stack(per_agent_rewards, axis=0)
    total_agents_rewards = per_agent_rewards.sum(axis=1)
    episodes = df.iloc[0].to_numpy(dtype=int)
    plt.plot(episodes, total_agents_rewards, color='g', label="policy network")
    plt.xlabel('# episodes')
    plt.ylabel('mean episode reward')
    plt.title('Mean of sum of rewards after each update:')
    plt.legend()
    plt.savefig("episode_reward.png")


plot_results(csv_path="src/net_configs/rewards.csv")
