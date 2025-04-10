# models.py
import torch
import torch.nn as nn

class SharedActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, actor_hidden=64, critic_dims=[128, 128]):
        super().__init__()

        # Actor shared encoder
        self.actor_shared = nn.Sequential(
            nn.Linear(obs_dim, actor_hidden),
            nn.ReLU()
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(actor_hidden, n_actions),
            nn.Softmax(dim=-1)
        )

        # Critic network (dynamically defined)
        critic_layers = []
        in_dim = obs_dim
        for dim in critic_dims:
            critic_layers.extend([nn.Linear(in_dim, dim), nn.ReLU()])
            in_dim = dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x):
        actor_features = self.actor_shared(x)
        action_probs = self.actor(actor_features)
        value = self.critic(x).squeeze(-1)
        return action_probs, value
