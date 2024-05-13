import torch
import torch.nn.functional as F
from utils import soft_update
from DDPG_agent import DDPG_agent
from MADDPG_agent import MADDPG_agent
from MASAC_agent import MASAC_agent
from SAC_agent import SAC_agent
from MATD3_agent import MATD3_agent
from TD3_agent import TD3_agent
from actor_critic import Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_agents(n_adv, n_good, agent_list, gamma, tau, lr, env, adv_agent_network, good_agent_network, adv_model,
               good_model, comb_crit, wd):
    agents = {}

    adv_critic = Critic(adv_agent_network["critic_input_size"], hidden=adv_agent_network["actor_n_hidden"]).to(device)
    good_critic = Critic(good_agent_network["critic_input_size"], hidden=good_agent_network["actor_n_hidden"]).to(
        device)
    for i in range(n_adv):
        agents.update({agent_list[i]: adv_model(gamma=gamma, tau=tau, env=env,
                                                in_features=adv_agent_network["actor_input_size"],
                                                in_features_Q=adv_agent_network["critic_input_size"], lr=lr,
                                                hidden=adv_agent_network["actor_n_hidden"], wd=wd)})
        if comb_crit:
            agents[agent_list[i]].critic = adv_critic
    for i in range(n_good):
        agents.update({agent_list[i + n_adv]: good_model(gamma=gamma, tau=tau, env=env,
                                                         in_features=good_agent_network["actor_input_size"],
                                                         in_features_Q=good_agent_network["critic_input_size"], lr=lr,
                                                         hidden=good_agent_network["actor_n_hidden"], critic=adv_critic,
                                                         wd=wd)})
        if comb_crit:
            agents.update({agent_list[i + n_adv]: good_model(gamma=gamma, tau=tau, env=env,
                                                             in_features=good_agent_network["actor_input_size"],
                                                             in_features_Q=good_agent_network["critic_input_size"],
                                                             lr=lr,
                                                             hidden=good_agent_network["actor_n_hidden"],
                                                             critic=good_critic)})
    return agents


class MARL_TRAINER(object):
    def __init__(self, gamma, tau, env, good_agent_network,
                 adv_agent_network, lr, num_adv, num_good, agent_list, adv_model, good_model, comb_crit, wd, grad_clip):
        self.gamma = gamma
        self.tau = tau
        self.env = env
        self.grad_clip = grad_clip
        self.agent_list = agent_list
        self.agents = get_agents(num_adv, num_good, agent_list, gamma, tau, lr, env, adv_agent_network,
                                 good_agent_network, adv_model, good_model, comb_crit, wd)

    def update(self, memory, agent_list, agent, agent_indices, BATCH_SIZE):

        index = memory[agent_indices[agent]].make_index(BATCH_SIZE)
        if isinstance(self.agents[agent], DDPG_agent):
            return self.update_DDPG(*(memory[agent_indices[agent]].sample_index(index)), agent)
        # collect replay sample from all agents
        elif isinstance(self.agents[agent], MADDPG_agent):
            obs_n = []
            obs_next_n = []
            act_n = []
            for agent_id in agent_list:
                obs, act, rew, obs_next, done = memory[agent_indices[agent_id]].sample_index(index)
                obs_n.append(obs)
                obs_next_n.append(obs_next)
                act_n.append(act)
            obs, act, rew, obs_next, done = memory[agent_indices[agent]].sample_index(index)
            self.update_MADDPG(rew=rew, done=done, obs_n=obs_n, obs_next_n=obs_next_n, act_n=act_n,
                               agent_list=agent_list, agent=agent, agent_indices=agent_indices)
        elif isinstance(self.agents[agent], SAC_agent):
            pass
        elif isinstance(self.agents[agent], MASAC_agent):
            pass
        elif isinstance(self.agents[agent], TD3_agent):
            pass
        elif isinstance(self.agents[agent], MATD3_agent):
            pass

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

    def update_MADDPG(self, rew, done, obs_n, obs_next_n, act_n, agent_list, agent, agent_indices):
        # train Q-net
        target_act_next_n = [
            self.agents[agent_id].act_update_target(obs_next_n[agent_indices[agent_id]]).detach() for agent_id in
            agent_list]
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
        act_agent_pol, logits = self.agents[agent].act_update(obs_n[agent_indices[agent]])
        self.agents[agent].actor_optimizer.zero_grad()
        act_n_with_agent_pol = [act_n[agent_indices[agent_id]] if agent_id != agent
                                else act_agent_pol
                                for agent_id in agent_list]
        crit_input = torch.cat([obs_n_cat, torch.cat(act_n_with_agent_pol, dim=1)], dim=1)
        policy_loss = -self.agents[agent].critic(crit_input)
        policy_loss = policy_loss.mean() + 1e-3 * (logits ** 2).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent].actor.parameters(), self.grad_clip)
        self.agents[agent].actor_optimizer.step()

        soft_update(self.agents[agent].actor_target, self.agents[agent].actor, self.tau)
        soft_update(self.agents[agent].critic_target, self.agents[agent].critic, self.tau)

        return policy_loss.item(), Q_loss.item()

    def update_SAC(self, ddpg_obs, ddpg_act, ddpg_rew, ddpg_obs_next, ddpg_done, agent):
        pass

    def update_MASAC(self, rew, done, obs_n, obs_next_n, act_n, agent_list, agent, agent_indices):
        pass

    def update_TD3(self, ddpg_obs, ddpg_act, ddpg_rew, ddpg_obs_next, ddpg_done, agent):
        pass

    def update_MATD3(self, rew, done, obs_n, obs_next_n, act_n, agent_list, agent, agent_indices):
        pass
