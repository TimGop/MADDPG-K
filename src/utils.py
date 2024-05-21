from collections import namedtuple
import random
import numpy as np
import torch


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward')
                        )


class ReplayMemory(object):
    def __init__(self, size):
        self._store = []
        self._max_size = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._store)

    def clear(self):
        self._store = []
        self._next_idx = 0

    def add(self, obs, act, rew, next_obs, done, knn_curr_list, knn_next_list):

        data = (obs, act, rew, next_obs, done, knn_curr_list, knn_next_list)

        if len(self._store) <= self._next_idx:
            self._store.append(data)

        else:
            self._store[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._max_size

    def _enc_sample(self, idxes):
        obs, act, rew, next_obs, dones, knn_curr_lists, knn_next_lists = [], [], [], [], [], [], []
        for i in idxes:
            data = self._store[i]
            obs_t, act_t, rew_t, next_obs_t, done, knn_curr_list, knn_next_list = data
            obs.append(obs_t)
            act.append(act_t)
            rew.append(rew_t)
            next_obs.append(next_obs_t)
            dones.append(done)
            knn_curr_lists.append(knn_curr_list)
            knn_next_lists.append(knn_next_list)
        return (torch.tensor(np.stack(obs), dtype=torch.float32), torch.tensor(np.stack(act), dtype=torch.float32),
                torch.tensor(np.stack(rew), dtype=torch.float32), torch.tensor(np.stack(next_obs), dtype=torch.float32),
                torch.tensor(np.stack(dones), dtype=torch.float32), np.stack(knn_curr_lists), np.stack(knn_next_lists))

    def make_index(self, batch_size):
        return [random.randint(0, len(self._store) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._max_size for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._enc_sample(idxes)

    def sample(self, batch_size):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._store))
        return self._enc_sample(idxes)

    def collect(self):
        return self.sample(-1)
