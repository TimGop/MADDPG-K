from actor_critic import Actor, Critic
import torch
import torch.nn as nn
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG_agent(object):
    def __init__(self, gamma, tau, env, in_features, in_features_Q, lr, hidden, wd, n_hidden_layers, activation,
                 critic=None, out_features_A=5):
        self.gamma = gamma
        self.tau = tau
        self.env = env

        if activation == "relu":
            activation = nn.functional.relu
        # Define the actor
        self.actor = Actor(in_features=in_features, out_features=out_features_A, hidden=hidden,
                           n_hidden_layers=n_hidden_layers, activation=activation).to(device)
        self.actor_target = Actor(in_features=in_features, out_features=out_features_A, hidden=hidden,
                                  n_hidden_layers=n_hidden_layers, activation=activation).to(device)

        # Define the critic
        self.critic = Critic(in_features_Q, hidden=hidden, n_hidden_layers=n_hidden_layers, activation=activation).to(
            device) if critic is None else critic
        self.critic_target = Critic(in_features_Q, hidden=hidden, n_hidden_layers=n_hidden_layers,
                                    activation=activation).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr, weight_decay=wd)  # optimizer for actor net
        # weight_decay=1e-8???
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr, weight_decay=wd)  # optimizer for critic net

        hard_update(self.critic_target, self.critic)  # make sure _ and target have same weights
        hard_update(self.actor_target, self.actor)  # make sure _ and target have same weights

    def act_update_target(self, select_action_State):
        return torch.softmax(self.actor_target(select_action_State), dim=1)

    def act_update(self, select_action_State):
        logits = self.actor(select_action_State)
        action = torch.softmax(logits, dim=1)
        return action, logits

    def act(self, select_action_State):
        with torch.no_grad():
            logits = self.actor(select_action_State)
            gumbel_noise = -torch.log(-torch.log(torch.rand(logits.size())))
            action = torch.softmax(logits + gumbel_noise, dim=1)
            return action
