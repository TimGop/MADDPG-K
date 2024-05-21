from actor_critic import Actor, Critic
import torch
from torch.optim import Adam
from utils import hard_update
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG_agent(object):
    def __init__(self, gamma, tau, env, in_features, in_features_Q, lr, hidden, wd, n_hidden_layers, activation,
                 critic=None, out_features_A=5):
        self.gamma = gamma
        self.tau = tau
        self.env = env

        # Define the actor
        if activation == "relu":
            activation = nn.functional.relu
        self.actor = Actor(in_features=in_features, out_features=out_features_A, hidden=hidden,
                           n_hidden_layers=n_hidden_layers, activation=activation).to(device)
        self.actor_target = Actor(in_features=in_features, out_features=out_features_A, hidden=hidden,
                                  n_hidden_layers=n_hidden_layers, activation=activation).to(device)
        self.out_features_actor = out_features_A

        # Define the critic
        self.critic = Critic(in_features_Q, hidden=hidden, n_hidden_layers=n_hidden_layers, activation=activation).to(
            device) if critic is None else critic
        self.critic_target = Critic(in_features_Q, hidden=hidden, n_hidden_layers=n_hidden_layers,
                                    activation=activation).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr, weight_decay=wd)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr, weight_decay=wd)

        hard_update(self.critic_target, self.critic)
        hard_update(self.actor_target, self.actor)

    def act_update_target(self, select_action_State):
        return torch.softmax(self.actor_target(select_action_State), dim=1)

    def act_update(self, select_action_State):
        logits = self.actor(select_action_State)
        action = torch.softmax(logits, dim=1)
        return action, logits

    def act(self, select_action_State):
        with torch.no_grad():
            logits = self.actor(select_action_State)
            gumbel_noise = -torch.log(-torch.log(torch.rand((1, self.out_features_actor))))
            action = torch.softmax(logits + gumbel_noise, dim=1)
            return action
