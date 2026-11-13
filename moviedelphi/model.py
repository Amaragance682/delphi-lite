"""Delphi for movie-rating histories.

Adapted from model.py of Delphi (https://github.com/gerstung-lab/Delphi, commit fb72166),
itself based on nanoGPT. Delphi's code is MIT-licensed, Copyright (c) 2024 Gerstung Lab;
see THIRD_PARTY_NOTICES.md.

Kept from Delphi: the GPT-2 blocks, the sinusoidal age encoding in place of position embeddings,
weight tying, the attention mask for padding, and the loss: cross-entropy for the next event plus
an exponential likelihood for the waiting time, whose rate is the sum of exp(logits).

Changed for whole-day movie data:
  * age_scale: the age encoding measures time in units of age_scale days (Delphi: 365.25, years).
  * Same-day ties: every target on day d is predicted from the last position whose day is
    strictly earlier than d. Delphi's mask_ties option removes same-day events from attention, but
    the current token still reaches its own prediction through the residual connection, so a
    same-day event could still be used. Reading the prediction from the last earlier-day position
    cannot see day d at all. Training and evaluation therefore use the same rule.
  * The reserved tokens (0 padding, 1 "no event") are never predicted.
  * generate() and the block-size helpers are not used here and were removed.
"""

import inspect
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

from .common import RESERVED_TOKENS


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, attn_mask):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # same computation as Delphi's explicit softmax(q k^T / sqrt(d)) with masked entries at -inf
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(new_gelu(self.c_fc(x))))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class AgeEncoding(nn.Module):
    """Delphi's sinusoidal encoding of event time, followed by a linear map."""

    def __init__(self, config):
        super().__init__()
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        self.register_buffer('div_term', div_term)
        self.n_embd = config.n_embd
        self.age_scale = config.age_scale
        self.linear = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(x / self.age_scale * self.div_term)
        y[..., 1::2] = torch.cos(x / self.age_scale * self.div_term)
        return self.linear(y)


@dataclass
class DelphiConfig:
    block_size: int = 256
    vocab_size: int = 10002
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    token_dropout: float = 0.0
    t_min: float = 0.1          # days; Delphi's floor on the expected waiting time
    bias: bool = False
    age_scale: float = 365.25   # days per unit of the age encoding (Delphi: years)
    basket_weighting: bool = False  # weight each day-to-next-day transition equally
    objective: str = "last_position"  # or "every_position", see Delphi.forward
    popularity_offset: bool = False  # add beta * log(1 + recent training-user count) to every log-rate
    popularity_window: int = 30
    ignore_tokens: list = field(default_factory=lambda: list(RESERVED_TOKENS))


def source_positions(idx, age, targets, targets_age):
    """For each target, the last input position on a strictly earlier day.

    Inputs are left-padded (token 0, age -10000) and sorted by day, so the positions with
    age < target day form a prefix; positions after i are never earlier than target i.
    Returns (src, valid): src is clamped to >= 0, valid marks real movie targets that have
    a real (non-padding) earlier-day position.
    """
    src = torch.searchsorted(age.contiguous(), targets_age.contiguous(), right=False) - 1
    src_c = src.clamp(min=0)
    valid = (src >= 0) & (targets > max(RESERVED_TOKENS)) & (idx.gather(1, src_c) > 0)
    return src_c, valid


def next_day_pairs(idx, age):
    """For every position, the events of the first later day that has events in the window.

    Returns pair (B, T, T) with pair[b, i, j] True if position j is on the next active day after
    position i, has_next (B, T) for real positions that have such a day, and next_day (B, T).
    """
    t = idx.size(1)
    real = idx > 0
    nxt = torch.searchsorted(age.contiguous(), age.contiguous(), right=True)  # first position with a later day
    has_next = real & (nxt < t)
    next_day = age.gather(1, nxt.clamp(max=t - 1))
    pair = (age[:, None, :] == next_day[:, :, None]) & real[:, None, :] & has_next[:, :, None]
    return pair, has_next, next_day


class Delphi(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wae=AgeEncoding(config),
            token_drop=nn.Dropout(config.token_dropout),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying, as in Delphi
        self.register_buffer('ignore', torch.tensor(config.ignore_tokens, dtype=torch.long), persistent=False)
        if config.popularity_offset:
            # baseline hazard: log(1 + training-user ratings of each movie in the window up to a day);
            # the table is rebuilt from the training split by set_popularity(), not stored in checkpoints
            self.pop_beta = nn.Parameter(torch.tensor(1.0))
            self.register_buffer('pop_log', torch.zeros(0, config.vocab_size), persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def hidden(self, idx, age):
        """Final hidden states (B, T, C) for left-padded token and age tensors."""
        device = idx.device
        t = idx.size(1)
        x = self.transformer.token_drop(self.transformer.wte(idx)) + self.transformer.wae(age.unsqueeze(-1))
        x = self.transformer.drop(x)

        # Delphi's mask: real positions attend to earlier real positions, padding only to itself
        real = idx > 0
        causal = torch.tril(torch.ones(t, t, device=device, dtype=torch.bool))[None, None]
        eye = torch.eye(t, device=device, dtype=torch.bool)[None, None]
        attn_mask = (real[:, None, None, :] & real[:, None, :, None] & causal) | (~real[:, None, None, :] & eye)

        for block in self.transformer.h:
            x = block(x, attn_mask)
        return self.transformer.ln_f(x)

    def set_popularity(self, table):
        """table: (n_days, vocab) float tensor of log(1 + windowed training-user counts)."""
        self.pop_log = table.to(self.lm_head.weight.device, torch.float16)

    def logits_from_hidden(self, h, day=None):
        """Log-rates (..., vocab). `day` (same leading shape as h) is the day each prediction is made at."""
        logits = self.lm_head(h).float()
        if self.config.popularity_offset:
            assert self.pop_log.shape[0] > 0, "call set_popularity() first"
            d = day.long().clamp(0, self.pop_log.shape[0] - 1)
            logits = logits + self.pop_beta * self.pop_log[d].float()
        return logits.index_fill(-1, self.ignore, float('-inf'))

    def target_logits(self, idx, age, targets, targets_age):
        """Log-rates for every valid target, each read at its last earlier-day position."""
        h = self.hidden(idx, age)
        src, valid = source_positions(idx, age, targets, targets_age)
        h_src = h.gather(1, src.unsqueeze(-1).expand(-1, -1, h.size(-1)))[valid]
        return self.logits_from_hidden(h_src, age.gather(1, src)[valid]), src, valid

    def forward(self, idx, age, targets, targets_age):
        """Training loss. targets[i] / targets_age[i] are the event after input position i.

        objective "last_position": every event is a target, predicted from the last position of
            the previous active day (one cross-entropy term per event).
        objective "every_position": every position predicts the whole basket of the next active
            day (cross-entropy averaged over the basket), so all positions receive a training signal.
        Either way a prediction for day d only depends on events from days before d.
        """
        if self.config.objective == "every_position":
            return self.forward_every_position(torch.cat([idx, targets[:, -1:]], 1),
                                               torch.cat([age, targets_age[:, -1:]], 1))
        logits, src, valid = self.target_logits(idx, age, targets, targets_age)
        y = targets[valid]
        if self.config.basket_weighting:
            # all targets of one day share a source position: give every such day the same total weight
            per_source = torch.zeros(src.shape, device=src.device).scatter_add_(1, src, valid.float())
            w = 1.0 / per_source.gather(1, src)[valid]
        else:
            w = torch.ones(len(y), device=y.device)
        w = w / w.sum()
        loss_ce = torch.sum(w * F.cross_entropy(logits, y, reduction='none'))

        # time from the last earlier-day event to the target, Delphi's exponential log-likelihood
        dt = (targets_age - age.gather(1, src))[valid]
        lse = torch.logsumexp(logits, -1)
        lse = -torch.log(torch.exp(-lse) + self.config.t_min)
        ldt = -torch.log(dt + self.config.t_min)
        loss_dt = torch.sum(w * -(lse - torch.exp(lse - ldt)))
        return {'loss_ce': loss_ce, 'loss_dt': loss_dt, 'n_targets': valid.sum()}

    def forward_every_position(self, idx, age):
        assert not self.config.popularity_offset, "not implemented for the every_position objective"
        logits = self.logits_from_hidden(self.hidden(idx, age))  # (B, T, V)
        pair, has_next, next_day = next_day_pairs(idx, age)
        logp = F.log_softmax(logits, -1)
        g = logp.gather(2, idx[:, None, :].expand(-1, idx.size(1), -1))  # g[b, i, j] = log p_i(movie at j)
        ce = -torch.where(pair, g, 0.0).sum(-1) / pair.sum(-1).clamp(min=1)
        if self.config.basket_weighting:
            same_day = (age[:, :, None] == age[:, None, :]) & has_next[:, None, :]
            w = has_next / same_day.sum(-1).clamp(min=1)
        else:
            w = has_next.float()
        w = w / w.sum()
        loss_ce = torch.sum(w * ce)

        dt = (next_day - age).clamp(min=1.0)
        lse = torch.logsumexp(logits, -1)
        lse = -torch.log(torch.exp(-lse) + self.config.t_min)
        ldt = -torch.log(dt + self.config.t_min)
        loss_dt = torch.sum(w * -(lse - torch.exp(lse - ldt)))
        return {'loss_ce': loss_ce, 'loss_dt': loss_dt, 'n_targets': has_next.sum()}

    @torch.no_grad()
    def predict_last(self, idx, age):
        """Log-rates for the next event after the last (right-most) position: (B, vocab)."""
        h = self.hidden(idx, age)
        return self.logits_from_hidden(h[:, -1, :], age[:, -1])

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Delphi/nanoGPT parameter groups: decay Linear weights only (the tied embedding is not decayed)."""
        decay, no_decay = set(), set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias') or fpn == 'pop_beta':
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, nn.Linear):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (nn.LayerNorm, LayerNorm, nn.Embedding)):
                    no_decay.add(fpn)
        decay.remove('lm_head.weight')
        param_dict = {pn: p for pn, p in self.named_parameters()}
        assert not (decay & no_decay) and not (param_dict.keys() - (decay | no_decay))
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        use_fused = device_type == 'cuda' and 'fused' in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **(dict(fused=True) if use_fused else {}))
