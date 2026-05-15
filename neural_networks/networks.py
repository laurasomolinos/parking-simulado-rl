import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """Red neuronal que aproxima la función Q.
    Entrada: estado (18 valores continuos)
    Salida: Q-value para cada acción (7 valores)
    """

    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)
    
    
    
class DuelingQNetwork(nn.Module):
    """Red neuronal con arquitectura Dueling.
    Tronco compartido -> dos cabezas: V(s) y A(s,a) -> Q(s,a)
    """

    def __init__(self, n_obs, n_actions):
        super().__init__()

        # Tronco compartido
        self.trunk = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Cabeza de valor: V(s) -> escalar
        self.value_head = nn.Linear(64, 1)

        # Cabeza de ventaja: A(s,a) -> un valor por acción
        self.advantage_head = nn.Linear(64, n_actions)

    def forward(self, x):
        features = self.trunk(x)
        V = self.value_head(features)                  # (batch, 1)
        A = self.advantage_head(features)              # (batch, n_actions)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q
