import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, q_dim=1024, r_dim=1024, latent_dim=256, hidden_dim=256):
        super(Encoder, self).__init__()
        self.input_dim = q_dim + r_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, q, r):
        x = torch.cat([q, r], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return h, mean, logvar

class ReasoningPolicy(nn.Module):
    def __init__(self, q_dim=1024, latent_dim=256, hidden_dim=256):
        super(ReasoningPolicy, self).__init__()

        self.fc1 = nn.Linear(q_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)


    def forward(self, q):
        h = F.relu(self.fc1(q))
        h = F.relu(self.fc2(h))
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        return h, mean, logvar

class Decoder(nn.Module):
    def __init__(self, q_dim=1024, latent_dim=256, hidden_dim=256):
        super(Decoder, self).__init__()
        self.input_dim = q_dim + latent_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, q_dim)

    def forward(self, q, z):
        x = torch.cat([q, z], dim=1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        r_emb = self.fc_out(h)
        return r_emb
    