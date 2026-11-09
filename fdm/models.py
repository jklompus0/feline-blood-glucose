"""PyTorch model families + a small trainer with early stopping and seed ensembling."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_LOW = math.log(80.0)
torch.set_num_threads(4)


class MLP(nn.Module):
    """Tabular features + learned cat embedding -> logit."""

    def __init__(self, n_feat, n_cats, hidden=64, dropout=0.2, emb=3, **_):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb)
        self.net = nn.Sequential(
            nn.Linear(n_feat + emb, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x, cat, **_):
        return {"logit": self.net(torch.cat([x, self.emb(cat)], 1)).squeeze(1)}


class HistoryGRU(nn.Module):
    """GRU over the previous k cycles' full glucose curves, fused with current features."""

    def __init__(self, n_feat, n_cats, step_dim, hidden=64, dropout=0.2, emb=3, **_):
        super().__init__()
        self.emb = nn.Embedding(n_cats, emb)
        self.gru = nn.GRU(step_dim, hidden // 2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(n_feat + emb + hidden // 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))

    def forward(self, x, cat, hist, **_):
        _, h = self.gru(hist)
        z = torch.cat([x, self.emb(cat), h[-1]], 1)
        return {"logit": self.head(z).squeeze(1)}


class GlucoseCurveNet(nn.Module):
    """Physiology-structured model.

    A network maps context (everything except `hours`) to curve parameters:
      depth d >= 0, nadir time tn in [1, 12], width w > 0, noise sigma > 0.
    Predicted log-BG at h hours after the shot:
      mu(h) = log_pre + offset - d * exp(-0.5 * ((h - tn) / w)^2)
    Trained on the actual BG values (Gaussian NLL in log space) plus BCE on
    P(BG < 80) = Phi((log 80 - mu) / sigma). Using the full continuous signal
    rather than only the binary label is what makes this data-efficient.
    """

    def __init__(self, n_feat, n_cats, hours_idx, log_pre_idx, hidden=64, dropout=0.2,
                 emb=3, **_):
        super().__init__()
        self.h_idx, self.p_idx = hours_idx, log_pre_idx
        self.emb = nn.Embedding(n_cats, emb)
        self.net = nn.Sequential(
            nn.Linear(n_feat - 1 + emb + 1, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 5))
        self.register_buffer("h_mu", torch.tensor(0.0))
        self.register_buffer("h_sd", torch.tensor(1.0))
        self.register_buffer("p_mu", torch.tensor(0.0))
        self.register_buffer("p_sd", torch.tensor(1.0))

    def set_scale(self, h_mu, h_sd, p_mu, p_sd):
        for n, v in zip(["h_mu", "h_sd", "p_mu", "p_sd"], [h_mu, h_sd, p_mu, p_sd]):
            getattr(self, n).fill_(float(v))

    def forward(self, x, cat, **_):
        hours = x[:, self.h_idx] * self.h_sd + self.h_mu          # back to raw hours
        log_pre = x[:, self.p_idx] * self.p_sd + self.p_mu
        keep = [i for i in range(x.shape[1]) if i != self.h_idx]
        z = torch.cat([x[:, keep], self.emb(cat), (hours / 12).unsqueeze(1)], 1)
        o = self.net(z)
        depth = F.softplus(o[:, 0] + 0.5)
        tn = 1 + 11 * torch.sigmoid(o[:, 1])
        width = 0.5 + F.softplus(o[:, 2] + 1.5)
        offset = 0.5 * torch.tanh(o[:, 3])
        sigma = 0.05 + F.softplus(o[:, 4] - 1.0)
        mu = log_pre + offset - depth * torch.exp(-0.5 * ((hours - tn) / width) ** 2)
        zscore = (LOG_LOW - mu) / sigma
        p = 0.5 * (1 + torch.erf(zscore / math.sqrt(2)))
        logit = torch.logit(p.clamp(1e-5, 1 - 1e-5))
        return {"logit": logit, "mu": mu, "sigma": sigma}


def _loss(model, out, y, log_bg):
    bce = F.binary_cross_entropy_with_logits(out["logit"], y)
    if "mu" in out:
        nll = (torch.log(out["sigma"]) + 0.5 * ((log_bg - out["mu"]) / out["sigma"]) ** 2).mean()
        return nll + bce
    return bce


def _batch(d, idx=None):
    if idx is None:
        return d
    return {k: v[idx] for k, v in d.items()}


def train_one(make, tr, va, seed, lr=3e-3, wd=1e-4, max_epochs=400, patience=40,
              fixed_epochs=None, batch=256):
    """Train with mini-batches. If `fixed_epochs` is set, train that many epochs with
    no validation (used for the final refit on train+val)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(tr["y"])
    best, best_state, best_ep, bad = float("inf"), None, 0, 0
    epochs = fixed_epochs or max_epochs
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = _batch(tr, perm[i:i + batch])
            out = model(**b)
            loss = _loss(model, out, b["y"], b["log_bg"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if fixed_epochs:
            continue
        model.eval()
        with torch.no_grad():
            vl = _loss(model, model(**va), va["y"], va["log_bg"]).item()
        if vl < best - 1e-4:
            best, best_state, best_ep, bad = vl, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep or epochs


@torch.no_grad()
def predict(model, d):
    model.eval()
    return torch.sigmoid(model(**d)["logit"]).numpy()
