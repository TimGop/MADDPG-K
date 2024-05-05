import torch
import torch.nn.functional as F
from utils import soft_update
from DDPG_agent import DDPG_agent
from MADDPG_agent import MADDPG_agent
from utils import Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_agents(n_adv, n_good, agent_list, gamma, tau, lr, env, adv_agent_network, good_agent_network, adv_model, good_model):
    agents = {}
    for i in range(n_adv):
        #TODO: change to be able to initialize DDPG agents aswell
        agents.update(
            {agent_list[i]: adv_model(gamma=gamma, tau=tau, env=env,
                                         in_features=adv_agent_network["actor_input_size"],
                                         in_features_Q=adv_agent_network["critic_input_size"], lr=lr,
                                         hidden=adv_agent_network["actor_n_hidden"])})
    for i in range(n_good):
        agents.update({agent_list[i + n_adv]: good_model(gamma=gamma, tau=tau, env=env,
                                                           in_features=good_agent_network["actor_input_size"],
                                                           in_features_Q=good_agent_network["critic_input_size"], lr=lr,
                                                           hidden=good_agent_network["actor_n_hidden"])})
    return agents


class MARL_TRAINER(object):
    def __init__(self, BATCH_SIZE, update_iter, gamma, tau, env, good_agent_network,
                 adv_agent_network, lr, num_adv, num_good, agent_list, max_iter_per_ep, update_freq, adv_model, good_model):
        self.BATCH_SIZE = BATCH_SIZE
        self.max_replay_buffer_len = BATCH_SIZE * max_iter_per_ep
        self.update_iter = update_iter
        self.gamma = gamma
        self.tau = tau
        self.env = env
        self.agent_list = agent_list
        self.update_freq = update_freq
        self.agents = get_agents(num_adv, num_good, agent_list, gamma, tau, lr, env, adv_agent_network,
                                 good_agent_network, adv_model, good_model)
        
    def update_DDPG(self, ddpg_obs,ddpg_act,ddpg_rew,ddpg_obs_next,ddpg_done, agent):

        # Get the actions and the state values to compute the targets
        next_action_batch = self.agents[agent].act_update_target(ddpg_obs_next)
        next_state_action_values = self.agents[agent].critic_target(torch.cat([ddpg_obs_next, next_action_batch.detach()], dim = 1))

        # Compute the target
        reward_batch = ddpg_rew.unsqueeze(1)
        done_batch = ddpg_done.unsqueeze(1)
        expected_values = reward_batch + (1 - done_batch.float()) * self.gamma * next_state_action_values

        # Update the critic network
        self.agents[agent].critic_optimizer.zero_grad()
        state_action_batch = self.agents[agent].critic(torch.cat([ddpg_obs, ddpg_act], dim = 1))
        # print("Critic expected values: ", expected_values)
        # print("Critic values: ", state_action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        # print("state_action_batch: ", state_action_batch)
        # print("expected_values: ", expected_values.detach())
        # print("val_loss vec: ", expected_values.detach()-state_action_batch)
        # print("val loss: ", value_loss)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent].critic.parameters(), 1)
        self.agents[agent].critic_optimizer.step()

        # Update the actor network
        self.agents[agent].actor_optimizer.zero_grad()
        # minus in front of q-val because we do gradient ascent
        state_actions, logits = self.agents[agent].act_update(ddpg_obs)
        # print("next_time_target: ", next_time_batch)
        # print("state_time_policy: ", state_time)
        policy_loss = -self.agents[agent].critic(torch.cat([ddpg_obs, state_actions], dim = 1)).mean() + 1e-3 * (logits**2).mean()
        # print("policy loss from critic for actor: ", policy_loss)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent].actor.parameters(), 1)
        self.agents[agent].actor_optimizer.step()

        # Update the target networks
        soft_update(self.agents[agent].actor_target, self.agents[agent].actor, self.tau)
        soft_update(self.agents[agent].critic_target, self.agents[agent].critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def update(self, memory, agent_list, agent, agent_indices, steps, BATCH_SIZE):
        # replay buffer not full enough
        if all([len(memory[agent_indices[agent_id]]) < self.max_replay_buffer_len for agent_id in agent_list]):
            return
        # update every 100 steps
        if not steps % self.update_freq == 0:
            return
        
        replay_sample_index = memory[agent_indices[agent]].make_index(BATCH_SIZE)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = replay_sample_index
        for agent_id in agent_list:
            obs, act, rew, obs_next, done = memory[agent_indices[agent_id]].sample_index(index)
            if agent_id == agent:
                ddpg_obs,ddpg_act,ddpg_rew,ddpg_obs_next,ddpg_done = obs, act, rew, obs_next, done 
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = memory[agent_indices[agent]].sample_index(index)

        if isinstance(self.agents[agent], DDPG_agent):
            return self.update_DDPG(ddpg_obs,ddpg_act,ddpg_rew,ddpg_obs_next,ddpg_done, agent)
            
            
        # train Q-net
        target_act_next_n = [
            self.agents[agent_id].act_gumbel_update_target(obs_next_n[agent_indices[agent_id]]).detach() for agent_id in
            agent_list]
        crit_targ_input = torch.cat([torch.cat(obs_next_n, dim=1), torch.cat(target_act_next_n, dim=1)], dim=1)
        target_q_next = self.agents[agent].critic_target(crit_targ_input)
        target_q = rew.unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1).float()) * target_q_next

        self.agents[agent].critic_optimizer.zero_grad()
        obs_n_cat = torch.cat(obs_n, dim=1)
        crit_input = torch.cat([obs_n_cat, torch.cat(act_n, dim=1)], dim=1)
        Q_loss = F.mse_loss(self.agents[agent].critic(crit_input), target_q.detach())
        Q_loss.backward()
        self.agents[agent].critic_optimizer.step()

        # train Policy
        act_agent_pol, logits = self.agents[agent].act_gumbel_update(obs_n[agent_indices[agent]])
        self.agents[agent].actor_optimizer.zero_grad()
        act_n_with_agent_pol = [act_n[agent_indices[agent_id]] if agent_id != agent
                                else act_agent_pol
                                for agent_id in agent_list]
        crit_input = torch.cat([obs_n_cat, torch.cat(act_n_with_agent_pol, dim=1)], dim=1)
        policy_loss = -self.agents[agent].critic(crit_input)
        policy_loss = policy_loss.mean() + 1e-3 *(logits**2).mean()
        policy_loss.backward()

        soft_update(self.agents[agent].actor_target, self.agents[agent].actor, self.tau)
        soft_update(self.agents[agent].critic_target, self.agents[agent].critic, self.tau)

        return policy_loss, Q_loss
