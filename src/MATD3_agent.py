from actor_critic import Actor, Critic
import torch
from torch.optim import Adam
from utils import hard_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MATD3_agent(object):
    def __init__(self, gamma, tau, env, in_features, in_features_Q, lr, hidden, critic = None):
        self.gamma = gamma
        self.tau = tau
        self.env = env

        # Define the actor
        self.actor = Actor(in_features=in_features, hidden=hidden).to(device)
        self.actor_target = Actor(in_features=in_features, hidden=hidden).to(device)

        # Define the critic
        self.critic = Critic(in_features_Q, hidden=hidden).to(device) if critic == None else critic
        self.critic_target = Critic(in_features_Q, hidden=hidden).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        hard_update(self.critic_target, self.critic)
        hard_update(self.actor_target, self.actor)

    def act_update_target(self, select_action_State):
        logits = self.actor_target(select_action_State)
        action = torch.softmax(logits, dim=1)
        return action

    def act_update(self, select_action_State):
        logits = self.actor(select_action_State)
        action = torch.softmax(logits, dim=1)
        return action, logits

    def act(self, select_action_State):
        with torch.no_grad():
            logits = self.actor(select_action_State)
            gumbel_noise = -torch.log(-torch.log(torch.rand((1, 5))))
            action = torch.softmax(logits + gumbel_noise, dim=1)
            return action
