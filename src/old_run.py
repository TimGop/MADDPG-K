import csv
import time
import argparse
import torch
import numpy as np
from pettingzoo.mpe import simple_tag_v3,simple_adversary_v3,simple_crypto_v3,simple_push_v3,simple_reference_v3,simple_speaker_listener_v4,simple_spread_v3,simple_v3,simple_world_comm_v3
from utils import ReplayMemory, Transition
from evaluation import evaluateNetwork
from DDPG_agent import DDPG

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
    parser.add_argument("--update-rate", type=int, default=100, help="update policies once every time this many environment steps are completed")
    parser.add_argument("--memory", type=int, default=1e5, help="size of replay buffer")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="net_configs\\", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="net_configs\\", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    return parser.parse_args()

def display(env,lr, gamma, n_episodes 
          ,good_agent_network,
          adv_agent_network, n_good, n_adv,
          tau, load_path):
    env.reset()
    agent_list = env.agents
    agents = {}
    for i in range(n_adv):
        agents.update({agent_list[i]: DDPG(gamma=gamma, tau=tau, env=env, in_features=adv_agent_network["actor_input_size"], 
                            in_features_Q=adv_agent_network["critic_input_size"], lr=lr, hidden=adv_agent_network["actor_n_hidden"])})
    for i in range(n_good):
        agents.update({agent_list[i+n_adv]: DDPG(gamma=gamma, tau=tau, env=env, in_features=good_agent_network["actor_input_size"], 
                            in_features_Q=good_agent_network["critic_input_size"], lr=lr, hidden= good_agent_network["actor_n_hidden"])})
    if load_path!= None:
        for agent_id in agent_list:
            print(agent_id + ":")
            print("imported net configs...")
            agents[agent_id].actor.load_state_dict(
                torch.load(load_path+"actor_" + agent_id + ".pt"))
            agents[agent_id].actor_target.load_state_dict(
                torch.load(load_path+"actor_" + agent_id + ".pt"))
            agents[agent_id].critic.load_state_dict(
                torch.load(load_path+"critic_" + agent_id + ".pt"))
            agents[agent_id].critic_target.load_state_dict(
                torch.load(load_path+"critic_" + agent_id + ".pt"))
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

def train(env, BATCH_SIZE, lr, gamma, n_episodes 
          ,good_agent_network,
          adv_agent_network, n_good, n_adv,update_iter,save_iter,
          tau,output_path, load_path, memory):
    env.reset()
    agent_list = env.agents

    memory = {a:ReplayMemory(int(memory)) for a in agent_list}
    agents = {}
    for i in range(n_adv):
        agents.update({agent_list[i]: DDPG(gamma=gamma, tau=tau, env=env, in_features=adv_agent_network["actor_input_size"], 
                            in_features_Q=adv_agent_network["critic_input_size"], lr=lr, hidden=adv_agent_network["actor_n_hidden"])})
    for i in range(n_good):
        agents.update({agent_list[i+n_adv]: DDPG(gamma=gamma, tau=tau, env=env, in_features=good_agent_network["actor_input_size"], 
                            in_features_Q=good_agent_network["critic_input_size"], lr=lr, hidden= good_agent_network["actor_n_hidden"])})
    
    if load_path!= None:
        for agent_id in agent_list:
            print(agent_id + ":")
            print("imported net configs...")
            agents[agent_id].actor.load_state_dict(
                torch.load(load_path+"actor_" + agent_id + ".pt"))
            agents[agent_id].actor_target.load_state_dict(
                torch.load(load_path+"actor_" + agent_id + ".pt"))
            agents[agent_id].critic.load_state_dict(
                torch.load(load_path+"critic_" + agent_id + ".pt"))
            agents[agent_id].critic_target.load_state_dict(
                torch.load(load_path+"critic_" + agent_id + ".pt"))
    episodeList = []
    rewardsList = []
    start = time.time()
    steps = 0
    # TRAINING
    for i_episode in range(n_episodes):
        env.reset()
        last_obs_and_act = {a:None for a in agent_list}
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if last_obs_and_act[agent] is not None:
                memory[agent].push(torch.tensor(last_obs_and_act[agent][0]), torch.tensor(last_obs_and_act[agent][1]),
                                torch.tensor(termination).reshape(1), torch.tensor(observation),
                                torch.tensor(reward).reshape(1))
            if termination or truncation:
                action = None
            else:
                action = agents[agent].act(torch.tensor(observation).unsqueeze(0)).squeeze()
            if action is not None:
                action = np.array(action.detach())  # convert tensor output of π-net to np array for env.step()
                last_obs_and_act[agent] = (observation, action)
            env.step(action)  # step switches to next agent!
            batches = []
            can_update = True
            for u_agent in agent_list:
                can_update = can_update and (len(memory[u_agent]) >= BATCH_SIZE)
            if can_update and steps % (update_iter*len(agent_list)) == 0:
                for u_agent in agent_list:
                    transitions = memory[u_agent].sample(BATCH_SIZE)
                    batch = Transition(*zip(*transitions))
                    batches.append(batch)
                for i_agent, u_agent in enumerate(agent_list):
                    value_loss, policy_loss = agents[u_agent].update(batches, i_agent)  # optimize network/s
            steps += 1

        if i_episode % save_iter == 0:
            end = time.time()
            print(f"Episode: {i_episode}, Time : {end-start}")
            episodeList, rewardsList = evaluateNetwork(episodeNumbers=episodeList, averageRewards=rewardsList,
                                                    currentEpisodeNumber=i_episode,
                                                    agents=agents, agent_list=agent_list)
            start = time.time()
            for agent_id in agent_list:
                torch.save(agents[agent_id].actor.state_dict(),
                        output_path+"actor_" + agent_id + ".pt")
                torch.save(agents[agent_id].critic.state_dict(),
                        output_path+"critic_" + agent_id + ".pt")

    with open(output_path+"rewards.csv", 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(episodeList)
        csv_writer.writerow(rewardsList)
    print('Completed training...')

if __name__ == '__main__':
    args = parse_args()
    
    env_dict = {"simple_tag_v3":simple_tag_v3,"simple_adversary_v3":simple_adversary_v3,
                "simple_crypto_v3":simple_crypto_v3,"simple_push_v3":simple_push_v3,
                "simple_reference_v3":simple_reference_v3,"simple_speaker_listener_v4":simple_speaker_listener_v4,
                "simple_spread_v3":simple_spread_v3,"simple_v3":simple_v3,"simple_world_comm_v3":simple_world_comm_v3}
    if args.num_good != None and args.num_adv != None:
        parallel_env = env_dict[args.scenario].env(continuous_actions=True, render_mode = "human" if args.display else None, num_good = args.num_good, num_adv = args.num_adv)
    elif args.num_good != None:
        parallel_env = env_dict[args.scenario].env(continuous_actions=True, render_mode = "human" if args.display else None, num_good = args.num_good)
    elif args.num_adv != None:
        parallel_env = env_dict[args.scenario].env(continuous_actions=True, render_mode = "human" if args.display else None, num_adv = args.num_adv)
    else:
        parallel_env = env_dict[args.scenario].env(continuous_actions=True, render_mode = "human" if args.display else None)
    num_good = 0
    num_adv = 0
    for s in parallel_env.possible_agents:
        if s.__contains__("adversary"):
            num_adv += 1
        elif s.__contains__("agent"):
            num_good += 1
    settings_good = {"simple_tag_v3" : {"actor_input_size": 14, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 19, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_adversary_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_crypto_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_push_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_reference_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_speaker_listener_v4" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_spread_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_world_comm_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                }
    settings_adv = {"simple_tag_v3" : {"actor_input_size": 16, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 21, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_adversary_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_crypto_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_push_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_reference_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_speaker_listener_v4" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_spread_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                "simple_world_comm_v3" : {"actor_input_size": 0, "actor_output_size": 5, "actor_n_layers" : args.num_layers, "actor_n_hidden" : args.num_hidden,
                                   "critic_input_size": 0, "critic_output_size": 1, "critic_n_layers" : args.num_layers, "critic_n_hidden" : args.num_hidden},
                }
    if args.display:
        display(env=parallel_env, lr= args.lr, gamma= args.gamma, tau= args.tau, n_episodes= int(args.num_episodes)
          ,good_agent_network=settings_good[args.scenario],adv_agent_network=settings_adv[args.scenario],
          n_good=num_good, n_adv= num_adv, load_path=args.load_dir if args.restore or args.display else None)
    else:
        train(env=parallel_env, BATCH_SIZE= args.batch_size, lr= args.lr, gamma= args.gamma, tau= args.tau, n_episodes= int(args.num_episodes)
          ,good_agent_network=settings_good[args.scenario],adv_agent_network=settings_adv[args.scenario],
          n_good=num_good, n_adv= num_adv, 
          update_iter=args.update_rate, save_iter=args.save_rate,
          output_path=args.save_dir, memory = args.memory, load_path=args.load_dir if args.restore else None)
   