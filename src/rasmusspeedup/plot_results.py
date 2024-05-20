import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


def plot_results(csv_path):
    df = pd.read_csv(csv_path, sep= ",", header= None)
    per_agent_rewards = df.iloc[1].to_numpy()
    per_agent_rewards = [np.fromstring(per_agent_rewards[i][2:-1], dtype=np.float32, sep=" ") for i in range(per_agent_rewards.shape[0])]
    for i, a in enumerate(per_agent_rewards):
        if a.shape[0] != 4:
            per_agent_rewards[i] = per_agent_rewards[i-1]
    per_agent_rewards = np.stack(per_agent_rewards , axis=0)
    total_agents_rewards = per_agent_rewards.sum(axis = 1)
    episodes = df.iloc[0].to_numpy(dtype=int)
    plt.plot(episodes, total_agents_rewards, color='g', label="policy network")
    plt.xlabel('# episodes')
    plt.ylabel('mean episode reward')
    plt.title('Mean of sum of rewards after each update:')
    plt.legend()
    plt.savefig("episode_reward.png")
    
def plot_all_tog(csv_paths):
    tar_list = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, sep= ",", header= None)
        per_agent_rewards = df.iloc[1].to_numpy()
        per_agent_rewards = [np.fromstring(per_agent_rewards[i][1:-1], dtype=np.float32, sep=" ") for i in range(per_agent_rewards.shape[0])]
        per_agent_rewards = np.stack(per_agent_rewards , axis=0)
        total_agents_rewards = per_agent_rewards.sum(axis = 1)
        episodes = df.iloc[0].to_numpy(dtype=int)
        tar_list.append(total_agents_rewards)
    tar_matrix = np.stack(tar_list)
    mean_rew = tar_matrix.mean(axis=0)
    std_rew = tar_matrix.std(axis=0)
    plt.plot(episodes, mean_rew, color='darkslategray', label="Our DDPG", linewidth=0.6)
    #plt.fill_between(episodes, mean_rew-std_rew, mean_rew+std_rew,facecolor='lightsteelblue',alpha=0.7, edgecolor = 'lightslategray', linewidth=0.5)
    eps = np.array([i*100 for i in range(1,501)])
    tot_rew = []
    for i in range(1,10):
        with open(f"src/DDPG/DDPGpaper{i}_rewards.pkl", 'rb') as f:
            rewards = np.array(pickle.load(f))
            tot_rew.append(rewards)
    rew_m = np.stack(tot_rew).mean(axis = 0)
    rew_std = np.stack(tot_rew).std(axis = 0)
    plt.plot(eps, rew_m, color='midnightblue', label="Paper DDPG", linewidth = 0.6)
    #plt.fill_between(eps, rew_m-rew_std, rew_m + rew_std,facecolor='lightskyblue',alpha=0.7, edgecolor = 'deepskyblue', linewidth=0.5)
    eps = np.array([i*100 for i in range(1,501)])
    tot_rew = []
    for i in range(1,10):
        with open(f"src/MADDPG/DDPGpaper{i}_rewards.pkl", 'rb') as f:
            rewards = np.array(pickle.load(f))
            tot_rew.append(rewards)
    rew_m = np.stack(tot_rew).mean(axis = 0)
    rew_std = np.stack(tot_rew).std(axis = 0)
    plt.plot(eps, rew_m, color='maroon', label="Paper MADDPG", linewidth = 0.6)
    #plt.fill_between(eps, rew_m-rew_std, rew_m + rew_std,facecolor='salmon',alpha=0.7, edgecolor = 'tomato', linewidth=0.5)
    plt.xlabel('# episodes')
    plt.ylabel('mean episode reward')
    plt.title('Mean sum of rewards per episode:')
    plt.legend()
    plt.savefig(f"combined_rewards_mean.png")
    
def plot_agent_avg(csv_paths):
    tar_list = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, sep= ",", header= None)
        per_agent_rewards = df.iloc[1].to_numpy()
        per_agent_rewards = [np.fromstring(per_agent_rewards[i][2:-1], dtype=np.float32, sep=" ") for i in range(per_agent_rewards.shape[0])]
        for i, a in enumerate(per_agent_rewards):
            if a.shape[0] != 4:
                per_agent_rewards[i] = per_agent_rewards[i-1]
        per_agent_rewards = np.stack(per_agent_rewards , axis=0)
        total_agents_rewards = per_agent_rewards.sum(axis = 1)
        eps = df.iloc[0].to_numpy(dtype=int)
        tar_list.append(per_agent_rewards)
    tar_matrix = np.stack(tar_list)
    rew_m = tar_matrix.mean(axis=0)
    rew_std = tar_matrix.std(axis=0)
    plt.plot(eps, rew_m[:,0], color='darkred', label="Adversary 1", linewidth = 0.6)
    plt.plot(eps, rew_m[:,1], color='darkred', label="Adversary 2", linewidth = 0.6)
    plt.plot(eps, rew_m[:,2], color='darkred', label="Adversary 3", linewidth = 0.6)
    plt.plot(eps, rew_m[:,3], color='darkgreen', label="Agent 1", linewidth = 0.6)
    plt.fill_between(eps, rew_m[:,0]-rew_std[:,0], rew_m[:,0]+rew_std[:,0],facecolor='salmon',alpha=0.7, edgecolor = 'tomato', linewidth=0.5)
    plt.fill_between(eps, rew_m[:,1]-rew_std[:,1], rew_m[:,1] + rew_std[:,1],facecolor='salmon',alpha=0.7, edgecolor = 'tomato', linewidth=0.5)
    plt.fill_between(eps, rew_m[:,2]-rew_std[:,2], rew_m[:,2]+ rew_std[:,2],facecolor='salmon',alpha=0.7, edgecolor = 'tomato', linewidth=0.5)
    plt.fill_between(eps, rew_m[:,3]-rew_std[:,3], rew_m[:,3] + rew_std[:,3],facecolor='lightgreen',alpha=0.7, edgecolor = 'lawngreen', linewidth=0.5)
    plt.xlabel('# episodes')
    plt.ylabel('mean episode reward')
    plt.title('Our DDPG - Mean per-agent rewards per episode')
    plt.legend()
    plt.savefig(f"finalDDPG_our_peragent.png")
    
def plot_results_averages(csv_paths):
    tar_list = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, sep= ",", header= None)
        per_agent_rewards = df.iloc[1].to_numpy()
        per_agent_rewards = [np.fromstring(per_agent_rewards[i][2:-1], dtype=np.float32, sep=" ") for i in range(per_agent_rewards.shape[0])]
        for i, a in enumerate(per_agent_rewards):
            if a.shape[0] != 4:
                per_agent_rewards[i] = per_agent_rewards[i-1]
        per_agent_rewards = np.stack(per_agent_rewards , axis=0)
        total_agents_rewards = per_agent_rewards.sum(axis = 1)
        episodes = df.iloc[0].to_numpy(dtype=int)
        tar_list.append(total_agents_rewards)
    tar_matrix = np.stack(tar_list)
    mean_rew = tar_matrix.mean(axis=0)
    std_rew = tar_matrix.std(axis=0)
    plt.plot(episodes, mean_rew, color='black', label="Our DDPG", linewidth=0.6)
    """plt.plot(episodes, mean_rew+std_rew, color='blue', label="policy network")
    plt.plot(episodes, mean_rew-std_rew, color='blue', label="policy network")"""
    plt.fill_between(episodes, mean_rew-std_rew, mean_rew+std_rew,facecolor='lightskyblue',alpha=0.7, edgecolor = 'deepskyblue', linewidth=0.5)
    plt.xlabel('# episodes')
    plt.ylabel('mean episode reward')
    plt.title('Mean sum of rewards per episode')
    plt.legend()
    #plt.show()
    plt.savefig("finalDDPG_ours.png")
    
    
plot_all_tog(["src/Final_DDPG_Ours/rewards10.csv","src/Final_DDPG_Ours/rewards9.csv","src/Final_DDPG_Ours/rewards8.csv","src/Final_DDPG_Ours/rewards7.csv","src/Final_DDPG_Ours/rewards5.csv","src/Final_DDPG_Ours/rewards1.csv","src/Final_DDPG_Ours/rewards6.csv","src/Final_DDPG_Ours/rewards2.csv","src/Final_DDPG_Ours/rewards3.csv","src/Final_DDPG_Ours/rewards4.csv"])
#plot_results_averages(["src/net_configs/rewards1.csv","src/net_configs/rewards2.csv","src/net_configs/rewards3.csv","src/net_configs/rewards4.csv","src/net_configs/rewards5.csv","src/net_configs/rewards6.csv","src/net_configs/rewards7.csv","src/net_configs/rewards8.csv","src/net_configs/rewards9.csv","src/net_configs/rewards10.csv","src/net_configs/rewards11.csv","src/net_configs/rewards12.csv"])


#plot_results(csv_path= "src/net_configs/rewards.csv")