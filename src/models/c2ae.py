import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import CONFIG

class Fd(nn.Module):
    def __init__(self, in_dim=16, H=256, out_dim=10, fin_act=None):
        super(Fd, self).__init__()
        self.fc1 = nn.Linear(in_dim, H)
        self.fc2 = nn.Linear(H, out_dim)
        self.fin_act = fin_act

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return self.fin_act(x) if self.fin_act else x

class Fx(nn.Module):
    def __init__(self, in_dim, H1=512, H2=256, out_dim=16):
        super(Fx, self).__init__()
        self.fc1 = nn.Linear(in_dim, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class Fe(nn.Module):
    def __init__(self, in_dim=10, H=512, out_dim=16):
        super(Fe, self).__init__()
        self.fc1 = nn.Linear(in_dim, H)
        self.fc2 = nn.Linear(H, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x

class C2AE(nn.Module):
    def __init__(self, Fx, Fe, Fd):
        super(C2AE, self).__init__()
        self.Fx = Fx
        self.Fe = Fe
        self.Fd = Fd
        self.alpha = CONFIG['c2ae']['alpha']
        self.beta = CONFIG['c2ae']['beta']
        self.emb_lambda = CONFIG['c2ae']['embedding_lambda']
        latent_dim = CONFIG['c2ae']['latent_dim']
        self.latent_I = torch.eye(latent_dim).to(CONFIG['device'])

    def forward(self, x, y=None):
        if self.training:
            fx_x = self.Fx(x)
            fe_y = self.Fe(y)
            fd_z = self.Fd(fe_y)
            return fx_x, fe_y, fd_z
        else:
            return self.predict(x)

    def predict(self, x):
        return self.Fd(self.Fx(x))

    def losses(self, fx_x, fe_y, fd_z, y):
        l_loss = self.latent_loss(fx_x, fe_y)
        c_loss = self.corr_loss(fd_z, y)
        return l_loss, c_loss

    def latent_loss(self, fx_x, fe_y):
        c1 = fx_x - fe_y
        c2 = fx_x.T @ fx_x - self.latent_I
        c3 = fe_y.T @ fe_y - self.latent_I
        return torch.trace(c1 @ c1.T) + self.emb_lambda * torch.trace(c2 @ c2.T + c3 @ c3.T)

    def corr_loss(self, preds, y):
        ones = (y == 1)
        zeros = (y == 0)
        ix_matrix = ones[:, :, None] & zeros[:, None, :]
        diff_matrix = torch.exp(-(preds[:, :, None] - preds[:, None, :]))
        losses = torch.flatten(diff_matrix * ix_matrix, start_dim=1).sum(dim=1)
        losses /= (ones.sum(dim=1) * zeros.sum(dim=1) + 1e-4)
        losses[losses == float('Inf')] = 0
        losses[torch.isnan(losses)] = 0
        return losses.sum()
