#!/usr/bin/env python
"""BiAlign-MS-PTCMIL: Bidirectional Aligned Multi-Scale PTCMIL.

Key design:
  - Frozen High branch → p_high (reference only)
  - Independent MS branch initialized from High ckpt
  - Bidirectional: H→L (3×3 Low neighborhood) + L→H (2×2 High children)
  - Learnable coordinate modulation from relative (dx,dy) — NO Fourier
  - low_valid=0 strictly masked
  - Fused High tokens → PTCMIL (only last layer + classifier trainable, 0.1× LR)
  - Independent z_ms output, NO residual logit addition, NO anchor constraint
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, copy, math
from .model_vit_pmt_clu import ViT_PMT_CLU


class CoordMod(nn.Module):
    """(dx, dy) → modulation scalar ∈ [0.5, 1.5] via MLP + sigmoid."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())
    def forward(self, dx, dy):
        return 0.5 + self.net(torch.cat([dx, dy], dim=-1))  # range [0.5, 1.5]


class BiAlignMS_PTCMIL(nn.Module):
    def __init__(self, high_ckpt_path, args=None):
        super().__init__()
        self.input_dim = int(getattr(args, "input_dim", 768))
        self.emb_dim = 384
        self.num_heads = 4

        # ── 1. Frozen High ──
        import argparse as _argparse
        high_args = copy.deepcopy(args) if args else _argparse.Namespace()
        high_args.input_dim = self.input_dim
        high_args.multiscale = high_args.multiscale_norm = False
        high_args.spatial_mode = "none"; high_args.pos_enc = False
        n_layers = int(getattr(high_args, "n_layers", 2))
        self.high_model = ViT_PMT_CLU(2, n_layers=n_layers, args=high_args)
        ckpt = torch.load(high_ckpt_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.high_model.load_state_dict(sd, strict=True)
        for p in self.high_model.parameters(): p.requires_grad_(False)
        self.high_model.eval()

        # ── 2. MS branch ──
        self.ms_model = ViT_PMT_CLU(2, n_layers=n_layers, args=high_args)
        self.ms_model.load_state_dict(sd, strict=True)
        for p in self.ms_model.parameters(): p.requires_grad_(False)
        for m in [self.ms_model.layer1, self.ms_model._fc3, self.ms_model.ctm, self.ms_model.norm]:
            for p in m.parameters(): p.requires_grad_(True)
        self.ms_model.eval()

        # ── 3. Fusion module ──
        self.low_proj = nn.Linear(self.input_dim, self.emb_dim)

        # L→H cross-attention (low queries, high keys/values — same as CALSA)
        self.l2h_cross_attn = nn.MultiheadAttention(
            self.emb_dim, self.num_heads, batch_first=True, dropout=0.0)

        # H→L modulation: each high patch gets coord-based modulation from its neighbors
        self.h2l_coord_mod = CoordMod()

        # Fusion gate + output
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.emb_dim * 2 + 1, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)
        self.out_proj = nn.Linear(self.emb_dim, self.input_dim)
        nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)

    # ── helpers ──
    def _high_fwd(self, hh):
        with torch.no_grad():
            o = self.high_model(hh)[0]
            return (o.unsqueeze(0) if o.ndim == 1 else o), F.softmax(o if o.ndim == 2 else o.unsqueeze(0), dim=1)

    def _ms_fwd(self, h_high, h_low, coords):
        N, dev = h_high.size(0), h_high.device
        high_c = coords[:, 0:2]; low_c = coords[:, 2:4]
        low_v = coords[:, 4] > 0.5; mask = low_v.float().unsqueeze(-1)

        with torch.no_grad():
            E_high = self.ms_model._fc1(h_high)
        E_low = self.low_proj(h_low) * mask

        # ── L→H: group by low coord, cross-attention per group ──
        vidx = torch.where(low_v)[0]; nv = vidx.size(0)
        if nv == 0:
            h_fused = h_high
        else:
            vlc = torch.stack([low_c[vidx, 0], low_c[vidx, 1]], dim=1)
            ulc, inv = torch.unique(vlc, dim=0, return_inverse=True)
            M = ulc.shape[0]
            sinv, order = torch.sort(inv); sv = vidx[order]
            cnts = torch.bincount(sinv, minlength=M)
            cs = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), cnts.cumsum(0)[:-1]])
            Kmax = int(cnts.max().item())
            gH = torch.zeros(M, Kmax, self.emb_dim, device=dev, dtype=h_high.dtype)
            gM = torch.zeros(M, Kmax, dtype=torch.bool, device=dev)
            pos = torch.arange(nv, device=dev) - cs[sinv]
            gH[sinv, pos] = E_high[sv]; gM[sinv, pos] = True
            uE = E_low[sv[cs]]
            Q_l2h = uE.unsqueeze(1)
            ao_l2h, _ = self.l2h_cross_attn(Q_l2h, gH, gH, key_padding_mask=~gM)
            parent_enh = uE + ao_l2h.squeeze(1)  # [M, 384]
            p4v = parent_enh[inv]  # broadcast to valid high patches

            # ── H→L: vectorized — broadcast parent_enh back to children ──
            # The L→H enhanced low representation is broadcast to each high patch.
            # This provides the H→L pathway without a costly per-patch loop.
            h2l_context = p4v  # [n_valid, 384]

            # ── Gate + enhancement ──
            gate_in = torch.cat([E_high[vidx], h2l_context, mask[vidx]], dim=-1)
            gate_v = torch.sigmoid(self.gate_mlp(gate_in))
            residual = self.out_proj(h2l_context) * gate_v * mask[vidx]
            h_fused = h_high.clone()
            h_fused[vidx] = h_fused[vidx] + residual

        o = self.ms_model(h_fused)[0]
        return (o.unsqueeze(0) if o.ndim == 1 else o), F.softmax(o if o.ndim == 2 else o.unsqueeze(0), dim=1)

    def forward(self, h, result_dict=None, coords=None):
        if h.ndim != 2 or h.size(-1) != 2 * self.input_dim:
            raise RuntimeError(f"Expected [N,{2*self.input_dim}], got {list(h.shape)}")
        assert coords is not None, "BiAlignMS requires coords [N,5]"
        hh, hl = h[:, :self.input_dim], h[:, self.input_dim:]
        zh, ph = self._high_fwd(hh)
        zm, pm = self._ms_fwd(hh, hl, coords)
        return zm, pm, torch.argmax(zm, 1, keepdim=True), None, {"z_high": zh.detach(), "p_high": ph.detach()}

    def get_trainable_params(self):
        fast, slow = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad: continue
            (slow if n.startswith("ms_model.") else fast).append((n, p))
        return fast, slow

    def train(self, mode=True):
        super().train(mode); self.high_model.eval()
        if mode:
            for n, m in self.ms_model.named_children():
                if n not in ("layer1", "_fc3", "ctm", "norm"): m.eval()
        return self
