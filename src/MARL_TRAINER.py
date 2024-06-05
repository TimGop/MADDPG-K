import torch
import torch.nn.functional as F
import numpy as np
from utils import soft_update
from DDPG_agent import DDPG_agent
from MADDPG_agent import MADDPG_agent
from actor_critic import Critic
import time

device = torch.device("cpu")


def get_agents(n_adv, n_adv_alt, n_good, agent_list, gamma, tau, lr, env, adv_agent_network, adv_alt_agent_network,
               good_agent_network, adv_model, adv_alt_model, good_model, comb_crit, wd):
    agents = {}
    if n_adv > 0 and comb_crit:
        adv_critic = Critic(adv_agent_network["critic_input_size"], hidden=adv_agent_network["actor_n_hidden"]).to(
            device)
    else:
        adv_critic = None

    if n_adv_alt > 0 and comb_crit:
        adv_alt_critic = Critic(adv_alt_agent_network["critic_input_size"],
                                hidden=adv_alt_agent_network["actor_n_hidden"]).to(device)
    else:
        adv_alt_critic = None

    good_critic = Critic(good_agent_network["critic_input_size"], hidden=good_agent_network["n_hidden"]).to(
        device)

    for i in range(n_adv_alt):
        if not comb_crit:
            agents.update({agent_list[i]: adv_alt_model(gamma=gamma, tau=tau, env=env,
                                                        in_features=adv_alt_agent_network["actor_input_size"],
                                                        in_features_Q=adv_alt_agent_network["critic_input_size"], lr=lr,
                                                        hidden=adv_alt_agent_network["n_hidden"], wd=wd,
                                                        out_features_A=adv_alt_agent_network["actor_output_size"],
                                                        n_hidden_layers=adv_alt_agent_network["n_layers"],
                                                        activation=adv_alt_agent_network["activation"])})
        else:
            # adv model instead
            agents.update({agent_list[i]: adv_alt_model(gamma=gamma, tau=tau, env=env, lr=lr, wd=wd,
                                                        in_features=adv_alt_agent_network["actor_input_size"],
                                                        in_features_Q=adv_alt_agent_network[
                                                            "critic_input_size"], critic=adv_alt_critic,
                                                        hidden=adv_alt_agent_network["n_hidden"],
                                                        out_features_A=adv_alt_agent_network[
                                                            "actor_output_size"],
                                                        n_hidden_layers=adv_alt_agent_network["n_layers"],
                                                        activation=adv_alt_agent_network["activation"])})
    for i in range(n_adv):
        if not comb_crit:
            agents.update({agent_list[i + n_adv_alt]: adv_model(gamma=gamma, tau=tau, env=env,
                                                                in_features=adv_agent_network["actor_input_size"],
                                                                in_features_Q=adv_agent_network["critic_input_size"],
                                                                lr=lr,
                                                                hidden=adv_agent_network["n_hidden"], wd=wd,
                                                                out_features_A=adv_agent_network["actor_output_size"],
                                                                n_hidden_layers=adv_agent_network["n_layers"],
                                                                activation=adv_agent_network["activation"])})
        else:
            # adv model instead
            agents.update({agent_list[i + n_adv_alt]: adv_model(gamma=gamma, tau=tau, env=env,
                                                                in_features=adv_agent_network["actor_input_size"],
                                                                in_features_Q=adv_agent_network["critic_input_size"],
                                                                lr=lr, wd=wd,
                                                                hidden=adv_agent_network["n_hidden"],
                                                                critic=adv_critic,
                                                                out_features_A=adv_agent_network["actor_output_size"],
                                                                n_hidden_layers=adv_agent_network["n_layers"],
                                                                activation=adv_agent_network["activation"])})
    for i in range(n_good):
        if not comb_crit:
            agents.update({agent_list[i + n_adv + n_adv_alt]: good_model(gamma=gamma, tau=tau, env=env,
                                                                         in_features=good_agent_network[
                                                                             "actor_input_size"],
                                                                         in_features_Q=good_agent_network[
                                                                             "critic_input_size"], lr=lr, wd=wd,
                                                                         hidden=good_agent_network["n_hidden"],
                                                                         out_features_A=good_agent_network[
                                                                             "actor_output_size"],
                                                                         n_hidden_layers=good_agent_network[
                                                                             "n_layers"],
                                                                         activation=good_agent_network[
                                                                             "activation"])})
        else:
            agents.update({agent_list[i + n_adv + n_adv_alt]: good_model(gamma=gamma, tau=tau, env=env,
                                                                         in_features=good_agent_network[
                                                                             "actor_input_size"],
                                                                         in_features_Q=good_agent_network[
                                                                             "critic_input_size"], lr=lr, wd=wd,
                                                                         hidden=good_agent_network["n_hidden"],
                                                                         critic=good_critic,
                                                                         out_features_A=good_agent_network[
                                                                             "actor_output_size"],
                                                                         n_hidden_layers=good_agent_network[
                                                                             "n_layers"],
                                                                         activation=good_agent_network[
                                                                             "activation"])})
    return agents


def get_i(a, a2, acts, obss, obss_next):
    return acts[a], obss[a], obss_next[a2]


def get_inds(array, array2, acts, obss, obss_next):
    vec_array_f = np.vectorize(get_i, signature="(),(),(l,n),(l,k),(l,k) -> (n),(k),(k)")
    obs_next_rp = np.tile(obss_next, (array.shape[0], 1, 1, 1)).transpose(0, 2, 1, 3)
    obs_rp = np.tile(obss, (array.shape[0], 1, 1, 1)).transpose(0, 2, 1, 3)
    acts_rp = np.tile(acts, (array.shape[0], 1, 1, 1)).transpose(0, 2, 1, 3)
    act, obs, obs_next = vec_array_f(array, array2, acts_rp, obs_rp, obs_next_rp)
    return act, obs, obs_next


class MARL_TRAINER(object):
    def __init__(self, gamma, tau, env, good_agent_network, adv_agent_network, adv_alt_agent_network, lr, num_adv,
                 num_good, agent_list, adv_model, adv_alt_model, good_model, comb_crit, wd, grad_clip, num_good_obs,
                 num_adv_obs, kNN_enabled, BATCH_SIZE, num_adv_alt, num_adv_alt_obs):
        self.start = None
        self.gamma = gamma
        self.tau = tau
        self.env = env
        self.grad_clip = grad_clip
        self.agent_list = agent_list
        self.agents = get_agents(num_adv, num_adv_alt, num_good, agent_list, gamma, tau, lr, env, adv_agent_network,
                                 adv_alt_agent_network, good_agent_network, adv_model, adv_alt_model, good_model,
                                 comb_crit, wd)
        self.kNN_enabled = kNN_enabled
        self.num_good_obs = num_good_obs
        self.num_adv_obs = num_adv_obs
        self.num_adv_alt_obs = num_adv_alt_obs
        self.BATCH_SIZE = BATCH_SIZE

    def update(self, memory, agent_list, agent, agent_indices, BATCH_SIZE):
        index = memory[agent_indices[agent]].make_index(BATCH_SIZE)
        if isinstance(self.agents[agent], DDPG_agent):
            return self.update_DDPG(*(memory[agent_indices[agent]].sample_index(index)[:5]), agent)
        # collect replay sample from all agents
        elif isinstance(self.agents[agent], MADDPG_agent):
            obs_n = []
            obs_next_n = []
            act_n = []
            for agent_id in agent_list:
                obs, act, _, obs_next, _, _, _ = memory[agent_indices[agent_id]].sample_index(index)
                obs_n.append(obs)
                obs_next_n.append(obs_next)
                act_n.append(act)

            if self.kNN_enabled:
                # knn_obs_lst/knn_obs_nxt_lst contains agent names/ids (knn for a given agent)
                _, _, _, _, _, knn_obs_lsts, knn_obs_nxt_lsts = memory[agent_indices[agent]].sample_index(index)
                obs_n_k_l = [[] for _ in range(self.num_adv_obs + self.num_good_obs + self.num_adv_alt_obs)]
                obs_next_n_k_l = [[] for _ in range(self.num_adv_obs + self.num_good_obs + self.num_adv_alt_obs)]
                act_n_k_l = [[] for _ in range(self.num_adv_obs + self.num_good_obs + self.num_adv_alt_obs)]
                obs_n_k = []
                obs_next_n_k = []
                act_n_k = []
                btch_idx = 0

                # TODO make code cleaner
                for knn_obs_lst in knn_obs_lsts:
                    knn_indices = {agent: knn_obs_lst.index(agent) for agent in knn_obs_lst}
                    for agent_id in knn_obs_lst:
                        bt = obs_n[agent_indices[agent_id]][btch_idx]
                        obs_n_k_l[knn_indices[agent_id]].append(obs_n[agent_indices[agent_id]][btch_idx])
                        act_n_k_l[knn_indices[agent_id]].append(act_n[agent_indices[agent_id]][btch_idx])
                    btch_idx += 1

                btch_idx = 0
                for knn_obs_nxt_lst in knn_obs_nxt_lsts:
                    knn_indices = {agent: knn_obs_nxt_lst.index(agent) for agent in knn_obs_nxt_lst}
                    for agent_id in knn_obs_nxt_lst:
                        obs_next_n_k_l[knn_indices[agent_id]].append(obs_next_n[agent_indices[agent_id]][btch_idx])
                    btch_idx += 1

                for i in range(len(obs_n_k_l)):
                    obs_n_k.append(torch.stack(obs_n_k_l[i]))
                    act_n_k.append(torch.stack(act_n_k_l[i]))
                    obs_next_n_k.append(torch.stack(obs_next_n_k_l[i]))
                obs_n = obs_n_k
                act_n = act_n_k
                obs_next_n = obs_next_n_k

            _, _, rew, _, done, _, _ = memory[agent_indices[agent]].sample_index(
                index)

            # only needed for pos of curr agent and to get its length in update maddpg if KNN active
            knn_indices = {agent_id: knn_obs_lsts[0].index(agent_id) for agent_id in
                           knn_obs_lsts[0]} if self.kNN_enabled else None
            return self.update_MADDPG(rew=rew, done=done, obs_n=obs_n, obs_next_n=obs_next_n, act_n=act_n,
                               agent_list=agent_list, agent=agent, agent_indices=agent_indices,
                               knn_obs_nxt_lsts=knn_obs_nxt_lsts, knn_indices=knn_indices)

    def update_DDPG(self, ddpg_obs, ddpg_act, ddpg_rew, ddpg_obs_next, ddpg_done, agent):

        # Get the actions and the state values to compute the targets
        next_action_batch = self.agents[agent].act_update_target(ddpg_obs_next)
        next_state_action_values = self.agents[agent].critic_target(
            torch.cat([ddpg_obs_next, next_action_batch.detach()], dim=1))

        # Compute the target
        reward_batch = ddpg_rew.unsqueeze(1)
        done_batch = ddpg_done.unsqueeze(1)
        expected_values = reward_batch + (1 - done_batch.float()) * self.gamma * next_state_action_values

        # Update the critic network
        self.agents[agent].critic_optimizer.zero_grad()
        state_action_batch = self.agents[agent].critic(torch.cat([ddpg_obs, ddpg_act], dim=1))
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent].critic.parameters(), self.grad_clip)
        self.agents[agent].critic_optimizer.step()

        # Update the actor network
        self.agents[agent].actor_optimizer.zero_grad()
        state_actions, logits = self.agents[agent].act_update(ddpg_obs)
        policy_loss = -self.agents[agent].critic(torch.cat([ddpg_obs, state_actions], dim=1)).mean() + 1e-3 * (
                logits ** 2).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent].actor.parameters(), self.grad_clip)
        self.agents[agent].actor_optimizer.step()

        # Update the target networks
        soft_update(self.agents[agent].actor_target, self.agents[agent].actor, self.tau)
        soft_update(self.agents[agent].critic_target, self.agents[agent].critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def update_MADDPG(self, rew, done, obs_n, obs_next_n, act_n, agent_list, agent, agent_indices, knn_obs_nxt_lsts
                      , knn_indices):
        # train Q-net
        if not self.kNN_enabled:
            target_act_next_n = [self.agents[agent_id].act_update_target(obs_next_n[agent_indices[agent_id]]).detach()
                                 for agent_id in agent_list]
        else:
            target_act_next_n = [[] for _ in range(len(knn_indices))]
            for batch_id in range(self.BATCH_SIZE):
                knn_nxt_indices = {agent_id: knn_obs_nxt_lsts[batch_id].index(agent_id) for agent_id in
                                   knn_obs_nxt_lsts[batch_id]}
                for i, agent_id in enumerate(knn_obs_nxt_lsts[batch_id]):
                    target_act_next_n[i].append(self.agents[agent_id].act_update_target(
                        obs_next_n[knn_nxt_indices[agent_id]][batch_id].unsqueeze(0)).detach())
            for k, targ_act in enumerate(target_act_next_n):
                target_act_next_n[k] = torch.cat(targ_act, dim=0)

        crit_targ_input = torch.cat([torch.cat(obs_next_n, dim=1), torch.cat(target_act_next_n, dim=1)], dim=1)
        target_q_next = self.agents[agent].critic_target(crit_targ_input)
        target_q = rew.unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1).float()) * target_q_next

        self.agents[agent].critic_optimizer.zero_grad()
        obs_n_cat = torch.cat(obs_n, dim=1)
        crit_input = torch.cat([obs_n_cat, torch.cat(act_n, dim=1)], dim=1)
        Q_loss = F.mse_loss(self.agents[agent].critic(crit_input), target_q.detach())
        Q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent].critic.parameters(), self.grad_clip)
        self.agents[agent].critic_optimizer.step()

        # train Policy
        if not self.kNN_enabled:
            act_agent_pol, logits = self.agents[agent].act_update(obs_n[agent_indices[agent]])
            self.agents[agent].actor_optimizer.zero_grad()
            act_n_with_agent_pol = [act_n[agent_indices[agent_id]] if agent_id != agent
                                    else act_agent_pol
                                    for agent_id in agent_list]
        else:
            act_agent_pol, logits = self.agents[agent].act_update(obs_n[knn_indices[agent]])
            self.agents[agent].actor_optimizer.zero_grad()
            act_n_with_agent_pol = act_n
            act_n_with_agent_pol[knn_indices[agent]] = act_agent_pol

        crit_input = torch.cat([obs_n_cat, torch.cat(act_n_with_agent_pol, dim=1)], dim=1)
        policy_loss = -self.agents[agent].critic(crit_input)
        policy_loss = policy_loss.mean() + 1e-3 * (logits ** 2).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent].actor.parameters(), self.grad_clip)
        self.agents[agent].actor_optimizer.step()

        soft_update(self.agents[agent].actor_target, self.agents[agent].actor, self.tau)
        soft_update(self.agents[agent].critic_target, self.agents[agent].critic, self.tau)

        return policy_loss.item(), Q_loss.item()