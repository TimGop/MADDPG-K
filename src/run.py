import csv
import time
import argparse
import torch
import numpy as np
from pettingzoo.mpe import simple_tag_v3, simple_adversary_v3, simple_crypto_v3, simple_push_v3, simple_reference_v3, \
    simple_speaker_listener_v4, simple_spread_v3, simple_v3, simple_world_comm_v3
from utils import ReplayMemory, Transition
from DDPG_agent import DDPG_agent
from MADDPG_agent import MADDPG_agent
from MARL_TRAINER import MARL_TRAINER

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_tag_v3", help="name of the scenario script")
    parser.add_argument("--num-episodes", type=int, default=5e4, help="number of episodes")
    parser.add_argument("--num-good", type=int, default=None, help="number of agents")
    parser.add_argument("--num-adv", type=int, default=None, help="number of adversaries")
    parser.add_argument("--good-agent", type=str, default="ddpg", help="policy for good agents")
    parser.add_argument("--adv-agent", type=str, default="ddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--tau", type=float, default=1e-2, help="target soft update parameter")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-hidden", type=int, default=64, help="number of hidden units in Network")
    parser.add_argument("--num-layers", type=int, default=1, help="number of hidden layers in Network")
    parser.add_argument("--update-rate", type=int, default=100, help="update policies once every time this many environment steps are completed (multiple of 25)")
    parser.add_argument("--memory", type=int, default=4e5, help="size of replay buffer")
    parser.add_argument("--bootstrap", type=bool, default=True, help="start training with random sampling")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./test_maddpg/", help="directory in which training state and model should be saved")
    parser.add_argument("--result-name", type=str, default="rewards.csv", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./test_maddpg/", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    return parser.parse_args()

def initialize_trainer(BATCH_SIZE, update_iter , gamma, tau, env,good_agent_network, adv_agent_network, lr, n_adv,
                    n_good, agent_list, max_iter_per_ep, good_model, adv_model, update_freq):
    return MARL_TRAINER(BATCH_SIZE=BATCH_SIZE, update_iter=update_iter, gamma=gamma, tau=tau, env=env,
                    good_agent_network=good_agent_network, adv_agent_network=adv_agent_network, lr=lr, num_adv=n_adv,
                    num_good=n_good, agent_list=agent_list, max_iter_per_ep=max_iter_per_ep, update_freq= update_freq, adv_model=adv_model, good_model=good_model)

#TODO: change display later
def display(env, lr, gamma, n_episodes, good_agent_network, adv_agent_network, n_good, n_adv, tau, load_path,adv_model,good_model):
    env.reset()
    agent_list = env.agents
    agents = {}
    for i in range(n_adv):
        agents.update(
            {agent_list[i]: DDPG_agent(gamma=gamma, tau=tau, env=env, in_features=adv_agent_network["actor_input_size"],
                                 in_features_Q=adv_agent_network["critic_input_size"], lr=lr,
                                 hidden=adv_agent_network["actor_n_hidden"])})
    for i in range(n_good):
        agents.update({agent_list[i + n_adv]: DDPG_agent(gamma=gamma, tau=tau, env=env,
                                                   in_features=good_agent_network["actor_input_size"],
                                                   in_features_Q=good_agent_network["critic_input_size"], lr=lr,
                                                   hidden=good_agent_network["actor_n_hidden"])})
    if load_path is not None:
        for agent in agent_list:
            print(agent + ":")
            print("imported net configs...")
            agents[agent].actor.load_state_dict(
                torch.load(load_path + "actor_" + agent + ".pt"))
            agents[agent].actor_target.load_state_dict(
                torch.load(load_path + "actor_" + agent + ".pt"))
            agents[agent].critic.load_state_dict(
                torch.load(load_path + "critic_" + agent + ".pt"))
            agents[agent].critic_target.load_state_dict(
                torch.load(load_path + "critic_" + agent + ".pt"))
    for i_episode in range(n_episodes):
        env.reset()
        for agent in env.agent_iter():
            env.render()
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = agents[agent].act(torch.tensor(observation).unsqueeze(0)).squeeze().numpy()
            env.step(action)  # step switches to next agent!


def train(env, BATCH_SIZE, lr, gamma, n_episodes, good_agent_network, adv_agent_network, update_iter, save_iter,
          tau, output_path, load_path, memory, result_name, adv_model, good_model, n_good, n_adv, bootstrap_sampling):
    env.reset()
    # TODO change to accommodate other envs
    max_iter_per_ep = 25
    agent_list = env.agents
    agent_indices = {agent: agent_list.index(agent) for agent in agent_list}
    memory = [ReplayMemory(int(memory)) for _ in agent_list]
    agent_trainer = initialize_trainer(BATCH_SIZE=BATCH_SIZE, update_iter=update_iter, gamma=gamma, tau=tau, env=env,
                    good_agent_network=good_agent_network, adv_agent_network=adv_agent_network, lr=lr, n_adv=n_adv,
                    n_good=n_good, agent_list=agent_list, max_iter_per_ep=max_iter_per_ep, good_model = good_model, adv_model = adv_model, update_freq = update_iter)

    if load_path is not None:
        for agent in agent_list:
            print(agent + ":")
            print("imported net configs...")
            agent_trainer.agents[agent].actor.load_state_dict(
                torch.load(load_path + "actor_" + agent + ".pt"))
            agent_trainer.agents[agent].actor_target.load_state_dict(
                torch.load(load_path + "actor_" + agent + ".pt"))
            agent_trainer.agents[agent].critic.load_state_dict(
                torch.load(load_path + "critic_" + agent + ".pt"))
            agent_trainer.agents[agent].critic_target.load_state_dict(
                torch.load(load_path + "critic_" + agent + ".pt"))
                
    episodeList = []
    rewardsList = []
    start = time.time()
    obs_n, _ = env.reset()
    steps = 0
    i_episode = 0
    ep_rew = np.zeros(shape=(len(agent_list)), dtype=np.int32)
    for i_episode in range(n_episodes):
        while True:
            if i_episode < BATCH_SIZE and bootstrap_sampling:
                action_n = {agent_id: env.action_space(agent_id).sample() for agent_id in agent_list}
            else:
                action_n = {agent: agent_trainer.agents[agent].act(torch.tensor(obs_n[agent]).unsqueeze(0)).squeeze().detach().numpy() for agent in agent_list}
            new_obs_n, rew_n, done_n, trunc, info_n = env.step(action_n)
            ep_rew += np.array([rew_n[agent] for agent in agent_list], dtype=np.int32)
            done = all(done_n.values()) or all(trunc.values())
            # collect experience
            for agent in agent_list:
                memory[agent_indices[agent]].add(torch.tensor(obs_n[agent]), torch.tensor(action_n[agent]),
                            torch.tensor(rew_n[agent]), torch.tensor(new_obs_n[agent]),torch.tensor(done_n[agent]))
            obs_n = new_obs_n
            for agent in agent_list:
                agent_trainer.update(memory, agent_list, agent, agent_indices, steps, BATCH_SIZE)
            if done:
                obs_n, _ = env.reset()
                break
            steps += 1
        if i_episode % save_iter == 0:
            end = time.time()
            print(f"Episode: {i_episode}, Time : {end-start}")
            ag_rew = ep_rew/save_iter
            ep_rew = np.zeros(shape=(len(agent_list)), dtype=np.int32)
            print(f"Reward : {ag_rew}, Total : {sum(ag_rew)}")
            episodeList.append(i_episode)
            rewardsList.append(ag_rew)
            start = time.time()
            for agent in agent_list:
                torch.save(agent_trainer.agents[agent].actor.state_dict(),
                           output_path + "actor_" + agent + ".pt")
                torch.save(agent_trainer.agents[agent].critic.state_dict(),
                           output_path + "critic_" + agent + ".pt")

    with open(output_path + result_name, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(episodeList)
        csv_writer.writerow(rewardsList)
    print('Completed training...')


if __name__ == '__main__':
    args = parse_args()
    algos = {"maddpg":MADDPG_agent,"ddpg":DDPG_agent}
    env_dict = {"simple_tag_v3":simple_tag_v3,"simple_adversary_v3":simple_adversary_v3,
                "simple_crypto_v3":simple_crypto_v3,"simple_push_v3":simple_push_v3,
                "simple_reference_v3":simple_reference_v3,"simple_speaker_listener_v4":simple_speaker_listener_v4,
                "simple_spread_v3":simple_spread_v3,"simple_v3":simple_v3,"simple_world_comm_v3":simple_world_comm_v3}
    if args.num_good != None and args.num_adv != None:
        parallel_env = env_dict[args.scenario].parallel_env(continuous_actions=True, render_mode = "human" if args.display else None, num_good = args.num_good, num_adv = args.num_adv)
    elif args.num_good != None:
        parallel_env = env_dict[args.scenario].parallel_env(continuous_actions=True, render_mode = "human" if args.display else None, num_good = args.num_good)
    elif args.num_adv != None:
        parallel_env = env_dict[args.scenario].parallel_env(continuous_actions=True, render_mode = "human" if args.display else None, num_adv = args.num_adv)
    else:
        parallel_env = env_dict[args.scenario].parallel_env(continuous_actions=True, render_mode = "human" if args.display else None)
    num_good = 0
    num_adv = 0
    sum_act_size = 0
    sum_obs_size = 0
    for s in parallel_env.possible_agents:
        sum_act_size += parallel_env.action_space(s).shape[0]
        sum_obs_size += parallel_env.observation_space(s).shape[0]
    for s in parallel_env.possible_agents:
        if s.__contains__("adversary"):
            num_adv += 1
            obs_sz =parallel_env.observation_space(s).shape[0]
            act_sz =parallel_env.action_space(s).shape[0]
            critic_input = obs_sz + act_sz if args.adv_agent == "ddpg" else sum_act_size + sum_obs_size
            settings_adv = {"actor_input_size": obs_sz, "actor_output_size": act_sz, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": critic_input, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden}
    
        elif s.__contains__("agent"):
            num_good += 1
            obs_sz =parallel_env.observation_space(s).shape[0]
            act_sz =parallel_env.action_space(s).shape[0]
            critic_input = obs_sz + act_sz if args.adv_agent == "ddpg" else sum_act_size + sum_obs_size
            settings_good = {"actor_input_size": obs_sz, "actor_output_size": act_sz, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": critic_input, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden}
    adv_model = algos[args.adv_agent]   
    good_model = algos[args.good_agent]   
    if args.display:
        display(env=parallel_env, lr= args.lr, gamma= args.gamma, tau= args.tau, n_episodes= int(args.num_episodes)
          ,good_agent_network=settings_good,adv_agent_network=settings_adv,
          n_good=num_good, n_adv= num_adv, load_path=args.load_dir if args.restore or args.display else None,adv_model=adv_model,good_model=good_model)
    else:
        train(env=parallel_env, BATCH_SIZE= args.batch_size, lr= args.lr, gamma= args.gamma, tau= args.tau, n_episodes= int(args.num_episodes)
          ,good_agent_network=settings_good,adv_agent_network=settings_adv,
          n_good=num_good, n_adv= num_adv, 
          update_iter=args.update_rate, save_iter=args.save_rate,
          output_path=args.save_dir, memory = args.memory, load_path=args.load_dir if args.restore else None,adv_model=adv_model,good_model=good_model, result_name=args.result_name, bootstrap_sampling=args.bootstrap)
   