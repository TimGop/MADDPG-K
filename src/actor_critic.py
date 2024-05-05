import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, in_features, hidden=64):
        super(Actor, self).__init__()
        self.input_layer = torch.nn.Linear(in_features=in_features, out_features=hidden)
        self.h0 = torch.nn.Linear(in_features=hidden, out_features=hidden)
        self.out_layer = torch.nn.Linear(in_features=hidden, out_features=5)

    def forward(self, f_state):
        x = f_state
        x.to(device)
        x = self.out_layer(
            torch.relu(self.h0(torch.relu(self.input_layer(x)))))
        return x


class Critic(nn.Module):
    def __init__(self, in_features_Q, hidden=64):
        super(Critic, self).__init__()
        self.input_layer = torch.nn.Linear(in_features=in_features_Q, out_features=hidden)
        self.h0 = torch.nn.Linear(in_features=hidden, out_features=hidden)
        self.out_layer = torch.nn.Linear(in_features=hidden, out_features=1)

    def forward(self, f_state):
        x = f_state
        x.to(device)
        x = self.out_layer(torch.relu(self.h0(torch.relu(self.input_layer(x)))))
        return x
