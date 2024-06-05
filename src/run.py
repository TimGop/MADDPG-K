import csv
import time
import argparse
import torch
import os
import numpy as np
from pettingzoo.mpe import simple_tag_v3, simple_adversary_v3, simple_push_v3, simple_reference_v3, \
    simple_speaker_listener_v4, simple_spread_v3, simple_v3, simple_world_comm_v3
from utils import ReplayMemory
from DDPG_agent import DDPG_agent
from MADDPG_agent import MADDPG_agent
from MARL_TRAINER import MARL_TRAINER
has_adv = False

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for MPE environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_push_v3", help="name of the scenario script",
                        choices=["simple_tag_v3", "simple_adversary_v3", "simple_spread_v3", "simple_v3",
                                 "simple_push_v3", "simple_reference_v3", "simple_speaker_listener_v4",
                                 "simple_world_comm_v3"])
    parser.add_argument("--num-episodes", type=int, default=int(5e4), help="number of episodes")
    parser.add_argument("--num-good", type=int, default=3, help="number of agents")
    parser.add_argument("--num-adv", type=int, default=1,
                        help="number of adversaries. If the environment allows for it")
    parser.add_argument("--num-adv-alt", type=int, default=1,
                        help="number of adversary alternatives (3rd agent type). If the environment allows for it")
    parser.add_argument("--num-good-obs", type=int, default=2,
                        help="number of good agents observed by other agents critics")
    parser.add_argument("--num-adv-obs", type=int, default=1,
                        help="number of adversaries observed by other agents critics")
    parser.add_argument("--num-adv-alt-obs", type=int, default=1,
                        help="number of adversary alternatives (3rd agent type) observed by other agents critics")
    parser.add_argument("--good-agent", type=str, default="maddpg", help="policy for good agents",
                        choices=["maddpg", "ddpg"])
    parser.add_argument("--adv-agent", type=str, default="maddpg", help="policy of adversaries",
                        choices=["maddpg", "ddpg"])
    parser.add_argument("--adv-alt-agent", type=str, default="maddpg",
                        help="policy of second adversary type or third agent", choices=["maddpg", "ddpg"])
    parser.add_argument("--kNN-enabled", type=bool, default=True, help="only look at kNN per critic")
    # Training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--tau", type=float, default=1e-2, help="target soft update parameter")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-hidden", type=int, default=64, help="number of hidden units in Network")
    parser.add_argument("--num-layers", type=int, default=1, help="number of hidden layers in Critic")
    parser.add_argument("--activation", type=str, default="relu", help="activation function")
    parser.add_argument("--update-rate", type=int, default=100,
                        help="update policies once every time this many environment steps are completed (multiple of "
                             "25)")
    parser.add_argument("--memory", type=int, default=int(1e5), help="size of replay buffer")
    parser.add_argument("--gradclip", type=float, default=1, help="Parameter norm gradient clipping")
    parser.add_argument("--wd", type=float, default=0, help="Optimizer weight decay")
    parser.add_argument("--bootstrap", type=bool, default=True, help="starting training with random sampling")
    parser.add_argument("--eps", type=float, default=0, help="epsilon exploration")
    parser.add_argument("--central-critic", type=bool, default=False, help="each group shares a central critic network")
    # Benchmarking
    parser.add_argument("--save-dir", type=str, default="./net_configs/")
    parser.add_argument("--result-name", type=str, default="rewards.csv",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./net_configs/",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    return parser.parse_args()


def get_knn(obs, agent, agent_list, num_good_obs, num_adv_obs, num_adv_alt_obs, scenario,good_start,adv_start):
    knn_list = []
    ag_id = agent_list.index(agent)
    if scenario == "simple_adversary_v3":
        dist = np.stack([obv[:2] for obv in list(obs.values())])
    else:
        dist = np.stack([obv[2:4] for obv in list(obs.values())])
    dist = np.linalg.norm(dist - dist[ag_id], axis= 1) + 1
    dist[ag_id] = 0
    adv_dist_ind = np.array([],dtype=np.int16)
    adv_alt_dist_ind = np.array([],dtype=np.int16)
    if num_adv_alt_obs > 0:
        adv_alt_dist_ind = dist[:adv_start].argsort(axis=0)[:num_adv_alt_obs]
    if num_adv_obs > 0:
        adv_dist_ind = dist[adv_start:good_start].argsort(axis=0)[:num_adv_obs] + num_adv_alt
    good_dist_ind = dist[good_start:].argsort(axis=0)[:num_good_obs] + num_adv_alt + num_adv
    knn_list = np.concatenate([adv_alt_dist_ind,adv_dist_ind,good_dist_ind], dtype=np.int16)
    ret_list = [agent_list[k] for k in knn_list]
    return ret_list


def initialize_trainer(gamma, tau, env, good_agent_network, adv_agent_network, adv_alt_agent_network, lr, n_adv,
                       n_adv_alt, n_good, agent_list, g_model, a_model, a_alt_model, comb_crit, wd, grad_clip,
                       num_good_obs=None, num_adv_obs=None,num_adv_alt_obs = None, kNN_enabled=False, BATCH_SIZE=None):
    return MARL_TRAINER(gamma=gamma, tau=tau, env=env, good_agent_network=good_agent_network,
                        adv_agent_network=adv_agent_network, adv_alt_agent_network=adv_alt_agent_network, lr=lr,
                        num_adv=n_adv, num_adv_alt=n_adv_alt, num_good=n_good, agent_list=agent_list, adv_model=a_model,
                        adv_alt_model=a_alt_model, good_model=g_model, comb_crit=comb_crit, wd=wd, grad_clip=grad_clip,
                        num_good_obs=num_good_obs, num_adv_alt_obs=num_adv_alt_obs,
                        num_adv_obs=num_adv_obs, kNN_enabled=kNN_enabled, BATCH_SIZE=BATCH_SIZE)


def display(env, lr, gamma, n_episodes, good_agent_network, adv_agent_network, adv_alt_agent_network,
            tau, load_path, a_model, a_alt_model, g_model,BATCH_SIZE,
          n_good, n_adv, n_adv_alt, comb_crit, wd, grad_clip, num_good_obs, num_adv_obs,kNN_enabled, num_adv_alt_obs):
    agent_list = env.possible_agents
    agent_trainer = initialize_trainer(gamma=gamma, tau=tau, env=env, good_agent_network=good_agent_network,
                                       adv_agent_network=adv_agent_network, adv_alt_agent_network=adv_alt_agent_network,
                                       lr=lr, n_adv=n_adv, n_adv_alt=n_adv_alt, n_good=n_good, agent_list=agent_list,
                                       g_model=g_model, a_model=a_model, a_alt_model=a_alt_model, comb_crit=comb_crit,
                                       wd=wd, grad_clip=grad_clip,num_good_obs=num_good_obs,num_adv_obs=num_adv_obs, kNN_enabled=kNN_enabled,
                                       num_adv_alt_obs = num_adv_alt_obs, BATCH_SIZE=BATCH_SIZE)
    if load_path is not None:
        for agent in agent_list:
            try:
                agent_trainer.agents[agent].actor.load_state_dict(
                    torch.load(load_path + "actor_" + agent + ".pt"))
                agent_trainer.agents[agent].actor_target.load_state_dict(
                    torch.load(load_path + "actor_" + agent + ".pt"))
                agent_trainer.agents[agent].critic.load_state_dict(
                    torch.load(load_path + "critic_" + agent + ".pt"))
                agent_trainer.agents[agent].critic_target.load_state_dict(
                    torch.load(load_path + "critic_" + agent + ".pt"))
            except RuntimeError:
                print("error trying to load network configs with wrong shape for current network")
                break

            print(agent + ":")
            print("imported net configs...")

    obs_n, _ = env.reset()
    for i_episode in range(n_episodes):
        while True:
            env.render()
            action = {agent: agent_trainer.agents[agent].act_update(torch.tensor(obs_n[agent]).unsqueeze(0))[
                0].squeeze().detach().numpy() for agent in agent_list}
            observation, reward, termination, truncation, info = env.step(action)
            obs_n = observation
            if all(termination.values()) or all(truncation.values()):
                obs_n, _ = env.reset()
                break


def train(env, BATCH_SIZE, lr, gamma, n_episodes, good_agent_network, adv_agent_network, adv_alt_agent_network,
          update_iter, save_iter, tau, output_path, load_path, memory, result_name, a_model, a_alt_model, g_model,
          n_good, n_adv, n_adv_alt, bootstrap_sampling, eps, comb_crit, wd, grad_clip, num_good_obs, num_adv_obs,
          num_adv_alt_obs, kNN_enabled, scenario):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    env.reset()

    agent_list = env.possible_agents
    adv_start = 0
    if scenario != "simple_speaker_listener_v4":
        if has_adv:
            adv_start = agent_list.index("adversary_0")
        good_start = agent_list.index("agent_0")
    else:
        adv_start = 0
        good_start = 0

    agent_indices = {agent: agent_list.index(agent) for agent in agent_list}
    memory = [ReplayMemory(int(memory)) for _ in agent_list]

    agent_trainer = initialize_trainer(gamma=gamma, tau=tau, env=env, good_agent_network=good_agent_network,
                                       adv_agent_network=adv_agent_network, adv_alt_agent_network=adv_alt_agent_network,
                                       lr=lr, n_adv=n_adv, n_adv_alt=n_adv_alt, n_good=n_good, agent_list=agent_list,
                                       g_model=g_model, a_model=a_model, a_alt_model=a_alt_model, comb_crit=comb_crit,
                                       wd=wd, grad_clip=grad_clip,num_good_obs=num_good_obs,num_adv_obs=num_adv_obs,num_adv_alt_obs=num_adv_alt_obs,
                                       kNN_enabled=kNN_enabled, BATCH_SIZE=BATCH_SIZE)

    if load_path is not None:
        for agent in agent_list:
            try:
                agent_trainer.agents[agent].actor.load_state_dict(
                    torch.load(load_path + "actor_" + agent + ".pt"))
                agent_trainer.agents[agent].actor_target.load_state_dict(
                    torch.load(load_path + "actor_" + agent + ".pt"))
                agent_trainer.agents[agent].critic.load_state_dict(
                    torch.load(load_path + "critic_" + agent + ".pt"))
                agent_trainer.agents[agent].critic_target.load_state_dict(
                    torch.load(load_path + "critic_" + agent + ".pt"))
            except RuntimeError:
                print("error trying to load network configs with wrong shape for current network")
                break

            print(agent + ":")
            print("imported net configs...")

    episodeList = []
    rewardsList = []
    start = time.time()
    obs_n, _ = env.reset()
    steps = 0
    ep_rew = np.zeros(shape=(len(agent_list)), dtype=np.int32)
    tme = 0
    lstms=0
    updtms =0
    for i_episode in range(n_episodes):
        while True:
            if bootstrap_sampling and (i_episode < BATCH_SIZE or eps >= np.random.rand()):
                action_n = {agent_id: env.action_space(agent_id).sample() for agent_id in agent_list}
            else:
                action_n = {agent: agent_trainer.agents[agent].act(
                    torch.tensor(obs_n[agent]).unsqueeze(0)).squeeze().detach().numpy() for agent in agent_list}
            new_obs_n, rew_n, done_n, trunc, info_n = env.step(action_n)
            ep_rew += np.array([rew_n[agent] for agent in agent_list], dtype=np.int32)
            done = all(done_n.values()) or all(trunc.values())
            # collect experience
            for agent in agent_list:
                # if knn enabled also precompute knn_list of obs for each agent for update
                knn_obs_lst = None
                knn_obs_nxt_lst = None
                if kNN_enabled:
                    strt = time.time_ns()
                    knn_obs_lst = get_knn(obs_n, agent, agent_list, num_good_obs, num_adv_obs, num_adv_alt_obs, scenario,good_start, adv_start)
                    knn_obs_nxt_lst = get_knn(new_obs_n, agent, agent_list, num_good_obs, num_adv_obs, num_adv_alt_obs, scenario,good_start, adv_start)
                    tme += (time.time_ns() - strt)
                memory[agent_indices[agent]].add(obs_n[agent], action_n[agent],
                                                 rew_n[agent], new_obs_n[agent],
                                                 done_n[agent], knn_obs_lst, knn_obs_nxt_lst)
            obs_n = new_obs_n
            if i_episode*25 > BATCH_SIZE and steps % update_iter == 0:
                for agent in agent_list:
                    lstm, uptm = agent_trainer.update(memory, agent_list, agent, agent_indices, BATCH_SIZE)
                    lstms += lstm
                    updtms += uptm
            if done:
                obs_n, _ = env.reset()
                break
            steps += 1
        if i_episode % save_iter == 0:
            end = time.time()
            print(f"Episode: {i_episode}, Time : {end - start}")
            ag_rew = ep_rew / save_iter
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
    algos = {"maddpg": MADDPG_agent, "ddpg": DDPG_agent}
    env_dict = {"simple_tag_v3": simple_tag_v3, "simple_adversary_v3": simple_adversary_v3,
                "simple_push_v3": simple_push_v3, "simple_reference_v3": simple_reference_v3,
                "simple_speaker_listener_v4": simple_speaker_listener_v4, "simple_spread_v3": simple_spread_v3,
                "simple_v3": simple_v3, "simple_world_comm_v3": simple_world_comm_v3}

    args.num_adv_alt_obs = min(args.num_adv_alt_obs,args.num_adv_alt)
    args.num_adv_obs = min(args.num_adv_obs,args.num_adv)
    args.num_good_obs = min(args.num_good_obs,args.num_good)
    if args.scenario == "simple_tag_v3":
        parallel_env = env_dict[args.scenario].parallel_env(continuous_actions=True,
                                                            render_mode="human" if args.display else None,
                                                            num_good=args.num_good, num_adversaries=args.num_adv)
        has_adv=True
        args.num_adv_alt = 0
        args.num_adv_alt_obs = 0
    elif args.scenario == "simple_spread_v3":
        parallel_env = env_dict[args.scenario].parallel_env(continuous_actions=True,
                                                            render_mode="human" if args.display else None,
                                                            N=args.num_good)
        
        args.num_adv_alt = 0
        args.num_adv_alt_obs = 0
        args.num_adv = 0
        args.num_adv_obs = 0
    elif args.scenario == "simple_adversary_v3":
        parallel_env = env_dict[args.scenario].parallel_env(continuous_actions=True,
                                                            render_mode="human" if args.display else None,
                                                            N=args.num_good)
        
        args.num_adv_alt = 0
        args.num_adv_alt_obs = 0
        args.num_adv = 1
        args.num_adv_obs = 1
        
    elif args.scenario == "simple_push_v3":
        parallel_env = env_dict[args.scenario].parallel_env(continuous_actions=True,
                                                            render_mode="human" if args.display else None)
        args.kNN_enabled = False  # fixed size small env --> KNN equivalent to MADDPG(unless only look at closest agent)
    elif args.scenario == "simple_v3":
        parallel_env = env_dict[args.scenario].parallel_env(continuous_actions=True,
                                                            render_mode="human" if args.display else None)
        args.kNN_enabled = False  # fixed size env with one agent --> KNN equivalent to MADDPG
    elif args.scenario == "simple_speaker_listener_v4":
        parallel_env = env_dict[args.scenario].parallel_env(continuous_actions=True,
                                                            render_mode="human" if args.display else None)
        args.kNN_enabled = False
    elif args.scenario == "simple_reference_v3":
        # local_ratio=0.5 is default (changing this value not of particular interest for our project)
        parallel_env = env_dict[args.scenario].parallel_env(local_ratio=0.5, continuous_actions=True,
                                                            render_mode="human" if args.display else None)
        args.kNN_enabled = False
    elif args.scenario == "simple_world_comm_v3":
        # default: num_obstacles=1, num_food=2, num_forests=2
        parallel_env = env_dict[args.scenario].parallel_env(num_good=args.num_good,
                                                            num_adversaries=args.num_adv+args.num_adv_alt,
                                                            num_obstacles=1, num_food=2, num_forests=2,
                                                            continuous_actions=True)
        has_adv=True
    else:
        raise Exception("The environment ", args.scenario, " is not implemented")

    parallel_env.metadata["render_fps"] = float(30)
    num_good = 0
    num_adv = 0
    num_adv_alt = 0
    sum_act_size = 0
    sum_obs_size = 0
    if args.kNN_enabled and args.adv_agent == "maddpg":
        # maddpg with knn
        n_good_obs = args.num_good_obs
        n_adv_obs = args.num_adv_obs
        n_adv_alt_obs = args.num_adv_alt_obs
        for s in parallel_env.possible_agents:
            if s.__contains__("agent") and n_good_obs > 0:
                sum_act_size += parallel_env.action_space(s).shape[0]
                sum_obs_size += parallel_env.observation_space(s).shape[0]
                n_good_obs -= 1
            if s.__contains__("adversary") and s.__contains__("lead") and n_adv_alt_obs > 0:
                sum_act_size += parallel_env.action_space(s).shape[0]
                sum_obs_size += parallel_env.observation_space(s).shape[0]
                n_adv_alt_obs -= 1

            elif s.__contains__("adversary") and n_adv_obs > 0:
                sum_act_size += parallel_env.action_space(s).shape[0]
                sum_obs_size += parallel_env.observation_space(s).shape[0]
                n_adv_obs -= 1
    else:
        for s in parallel_env.possible_agents:
            sum_act_size += parallel_env.action_space(s).shape[0]
            sum_obs_size += parallel_env.observation_space(s).shape[0]
    settings_adv = None
    settings_adv_alt = None
    settings_good = None
    for s in parallel_env.possible_agents:
        # print(s)
        if s.__contains__("adversary") and s.__contains__("lead"):
            num_adv_alt += 1
            obs_sz = parallel_env.observation_space(s).shape[0]
            act_sz = parallel_env.action_space(s).shape[0]
            critic_input = obs_sz + act_sz if args.adv_agent == "ddpg" else sum_act_size + sum_obs_size
            settings_adv_alt = {"actor_input_size": obs_sz, "actor_output_size": act_sz,
                                "n_layers": args.num_layers, "n_hidden": args.num_hidden,
                                "critic_input_size": critic_input, "critic_output_size": 1,
                                "activation": args.activation}
        elif s.__contains__("adversary"):
            num_adv += 1
            obs_sz = parallel_env.observation_space(s).shape[0]
            act_sz = parallel_env.action_space(s).shape[0]
            critic_input = obs_sz + act_sz if args.adv_agent == "ddpg" else sum_act_size + sum_obs_size
            settings_adv = {"actor_input_size": obs_sz, "actor_output_size": act_sz, "n_layers": args.num_layers,
                            "n_hidden": args.num_hidden,
                            "critic_input_size": critic_input, "critic_output_size": 1,
                            "activation": args.activation}
        elif s.__contains__("agent"):
            num_good += 1
            obs_sz = parallel_env.observation_space(s).shape[0]
            act_sz = parallel_env.action_space(s).shape[0]
            critic_input = obs_sz + act_sz if args.good_agent == "ddpg" else sum_act_size + sum_obs_size
            settings_good = {"actor_input_size": obs_sz, "actor_output_size": act_sz, "n_layers": args.num_layers,
                             "n_hidden": args.num_hidden,
                             "critic_input_size": critic_input, "critic_output_size": 1,
                             "activation": args.activation}

        elif s.__contains__("listener"):
            num_good += 1
            obs_sz = parallel_env.observation_space(s).shape[0]
            act_sz = parallel_env.action_space(s).shape[0]
            # print(obs_sz)
            # print(act_sz)
            critic_input = obs_sz + act_sz if args.good_agent == "ddpg" else sum_act_size + sum_obs_size
            settings_good = {"actor_input_size": obs_sz, "actor_output_size": act_sz, "n_layers": args.num_layers,
                             "n_hidden": args.num_hidden,
                             "critic_input_size": critic_input, "critic_output_size": 1,
                             "activation": args.activation}

        elif s.__contains__("speaker"):
            num_adv += 1
            obs_sz = parallel_env.observation_space(s).shape[0]
            act_sz = parallel_env.action_space(s).shape[0]
            # print(obs_sz)
            # print(act_sz)
            critic_input = obs_sz + act_sz if args.adv_agent == "ddpg" else sum_act_size + sum_obs_size
            settings_adv = {"actor_input_size": obs_sz, "actor_output_size": act_sz, "n_layers": args.num_layers,
                            "n_hidden": args.num_hidden,
                            "critic_input_size": critic_input, "critic_output_size": 1,
                            "activation": args.activation}
        else:
            raise Exception("Settings for agents could not be initialized (invalid agent types)...")

    adv_model = algos[args.adv_agent]
    good_model = algos[args.good_agent]
    adv_alt_model = algos[args.adv_alt_agent]
    if args.display:
        display(env=parallel_env, lr=args.lr, gamma=args.gamma, tau=args.tau,BATCH_SIZE=args.batch_size,
                n_episodes=int(args.num_episodes), good_agent_network=settings_good, adv_agent_network=settings_adv,
                adv_alt_agent_network=settings_adv_alt, n_good=num_good, n_adv=num_adv, n_adv_alt=num_adv_alt,
                load_path=args.load_dir, a_model=adv_model, a_alt_model=adv_alt_model, g_model=good_model,
                comb_crit=args.central_critic, wd=args.wd, grad_clip=args.gradclip,num_good_obs=args.num_good_obs,
              num_adv_obs=args.num_adv_obs, num_adv_alt_obs=args.num_adv_alt_obs, kNN_enabled=args.kNN_enabled)
    else:
        train(env=parallel_env, BATCH_SIZE=args.batch_size, lr=args.lr, gamma=args.gamma, tau=args.tau,
              n_episodes=int(args.num_episodes), good_agent_network=settings_good, adv_agent_network=settings_adv,
              adv_alt_agent_network=settings_adv_alt, n_good=num_good, n_adv=num_adv, n_adv_alt=num_adv_alt,
              update_iter=args.update_rate, save_iter=args.save_rate, output_path=args.save_dir, memory=args.memory,
              load_path=args.load_dir if args.restore else None, a_model=adv_model, a_alt_model=adv_alt_model,
              g_model=good_model, result_name=args.result_name, bootstrap_sampling=args.bootstrap, eps=args.eps,
              comb_crit=args.central_critic, wd=args.wd, grad_clip=args.gradclip, num_good_obs=args.num_good_obs,
              num_adv_obs=args.num_adv_obs, num_adv_alt_obs=args.num_adv_alt_obs, kNN_enabled=args.kNN_enabled,
              scenario=args.scenario)
