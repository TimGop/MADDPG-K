import torch
import torch.nn as nn

device = torch.device("cpu")


class Actor(nn.Module):
    def __init__(self, in_features, out_features, hidden=64, n_hidden_layers=1, activation=nn.functional.relu):
        super(Actor, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=in_features, out_features=hidden)])
        for _ in range(n_hidden_layers):
            self.layers.append(nn.Linear(in_features=hidden, out_features=hidden))
        self.out_layer = nn.Linear(in_features=hidden, out_features=out_features)
        self.activation = activation

    def forward(self, f_state):
        x = f_state
        x.to(device)
        for layer in self.layers:
            x = self.activation(layer(x))

        x = self.out_layer(x)
        return x


class Critic(nn.Module):
    def __init__(self, in_features_Q, hidden=64, n_hidden_layers=1, activation=nn.functional.relu):
        super(Critic, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=in_features_Q, out_features=hidden)])
        for _ in range(n_hidden_layers):
            self.layers.append(nn.Linear(in_features=hidden, out_features=hidden))
        self.out_layer = nn.Linear(in_features=hidden, out_features=1)
        self.activation = activation

    def forward(self, f_state):
        x = f_state
        x.to(device)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.out_layer(x)
        return x
