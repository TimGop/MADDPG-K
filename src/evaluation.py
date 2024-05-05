import torch
import numpy as np
from pettingzoo.mpe import simple_tag_v3

env = simple_tag_v3.env(continuous_actions=True)
num_of_tests = 20


def add_reward(reward, agent_name, rewardTotals):
    if agent_name == "adversary_0":
        rewardTotals[0] += reward
    elif agent_name == "adversary_1":
        rewardTotals[1] += reward
    elif agent_name == "adversary_2":
        rewardTotals[2] += reward
    elif agent_name == "agent_0":
        rewardTotals[3] += reward
    else:
        print("invalid name...")


def evaluateNetwork(episodeNumbers, averageRewards, currentEpisodeNumber, agents, agent_list):
    for agent in agent_list:
        agents[agent].set_eval()
    rewardTotals = np.array([0, 0, 0, 0])
    episodeNumbers.append(currentEpisodeNumber)

    for task_i_idx in range(num_of_tests):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            action = agents[agent].act(torch.tensor(observation).unsqueeze(0)).squeeze().numpy()
            if termination or truncation:
                action = None
            env.step(action)
            add_reward(reward, agent, rewardTotals)

    averageRewards.append((rewardTotals / num_of_tests))
    print(f"Per-agent mean reward: {averageRewards[-1]}")
    for agent in agent_list:
        agents[agent].set_train()
    return episodeNumbers, averageRewards
