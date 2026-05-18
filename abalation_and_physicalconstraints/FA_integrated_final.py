"""
FA_integrated_final.py — Comprehensive Single-Region Experiment Pipeline

Integrates all findings from V1–V7:
  - Architecture ablation: MLP, CNN, SA, ES-MHSA, FA, FA+LSF, FA+LSF+CCSI, FA+MSF, FA+MSF+CCSI
  - Explicit physics constraints: MGC, PFV, SER, All (adaptive weighting)
  - Implicit physics exploration: MTL, SPI, PFA, combinations
  - Multi-seed statistical evaluation
  - Complete data saving for downstream figure generation

Usage:
  python FA_integrated_final.py --region xisha --ws 21 --seeds 42 123 7 --outdir results_xisha
  python FA_integrated_final.py --region nanhai --ws 21 --seeds 42 123 7 --outdir results_nanhai

All results (metrics, training curves, predictions, model weights) are saved to --outdir.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os, json, argparse, time, pickle
from tqdm import tqdm
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================
# CONFIG (editable per region)
# ============================================================
FILE_MAP = {
    "xisha": {
        "ga": "../SWOTxisha.npy", "vgg": "../SWOTVGGxisha.npy",
        "ssh": "../SWOTDOVnorthxisha.npy", "dov": "../SWOTDOVeastxisha.npy",
        "gebco": "../b_xisha.npy", "truth": "../xishatrue.npy",
    },

    "pac": {
        "ga": "../SWOTpac.npy", "vgg": "../SWOTVGGpac.npy",
        "ssh": "../SWOTDOVnorthpac.npy", "dov": "../SWOTDOVeastpac.npy",
        "gebco": "../b_pac.npy", "truth": "../pactrue2.npy",
    },

    "carribean": {
        "ga": "../SWOTcarribean.npy", "vgg": "../SWOTVGGcarribean.npy",
        "ssh": "../SWOTDOVnorthcarribean.npy", "dov": "../SWOTDOVeastcarribean.npy",
        "gebco": "../b_carribean.npy", "truth": "../carribeantrue2.npy",
    },

    "mexico": {
        "ga": "../SWOTmexico.npy", "vgg": "../SWOTVGGmexico.npy",
        "ssh": "../SWOTDOVnorthmexico.npy", "dov": "../SWOTDOVeastmexico.npy",
        "gebco": "../b_mexico.npy", "truth": "../mexicotrue2.npy",
    },
    # Add more regions here:
    # "nanhai": { "ga": "...", ... },
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--region', type=str, default='xisha')
    p.add_argument('--ws', type=int, default=21)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 7])
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--outdir', type=str, default='results')
    p.add_argument('--skip_arch', action='store_true', help='Skip architecture ablation')
    p.add_argument('--skip_phys', action='store_true', help='Skip explicit physics')
    p.add_argument('--skip_implicit', action='store_true', help='Skip implicit physics')
    return p.parse_args()

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ============================================================
# 1. DATA
# ============================================================
def _w(d, r, c, ws):
    h = ws // 2; p = np.pad(d, h, mode='mean'); return p[r:r+ws, c:c+ws]

def load_data(region, ws=21, test_split=0.2):
    """Load and prepare data. Returns train/test splits + aux targets for MTL."""
    fm = FILE_MAP[region]
    logging.info(f"Loading data for region={region}, ws={ws}")
    ga = np.load(fm['ga']).squeeze(); vgg = np.load(fm['vgg']).squeeze()
    ssh = np.load(fm['ssh']).squeeze(); dov = np.load(fm['dov']).squeeze()
    geb = np.load(fm['gebco']); mb = np.load(fm['truth'])
    assert ga.shape == vgg.shape == ssh.shape == dov.shape
    ds = ga.shape; logging.info(f"Shape: {ds}")

    for arr in [ga, vgg, ssh, dov, geb]:
        arr[np.isnan(arr)] = np.nanmean(arr)

    tr = mb[::4, ::4]; gl = geb[::4, ::4]; mk = ~np.isnan(tr)
    assert mk.shape == ds; logging.info(f"Valid pixels: {mk.sum()}")

    c = ws // 2
    X, y, y_aux, pos = [], [], [], []
    for i in range(ds[0]):
        for j in range(ds[1]):
            if mk[i, j]:
                ga_w = _w(ga,i,j,ws); vgg_w = _w(vgg,i,j,ws)
                ssh_w = _w(ssh,i,j,ws); dov_w = _w(dov,i,j,ws); gl_w = _w(gl,i,j,ws)
                feat = np.concatenate([ga_w.flatten(), vgg_w.flatten(),
                    ssh_w.flatten(), dov_w.flatten(), gl_w.flatten()])
                X.append(feat); y.append(tr[i,j]); pos.append([i,j])
                y_aux.append([ga_w[c,c], vgg_w[c,c]])

    X, y, y_aux, pos = np.array(X), np.array(y), np.array(y_aux), np.array(pos)
    logging.info(f"Samples: {len(X)}, features: {X.shape[1]}")

    idx = np.arange(len(X))
    ti, ei = train_test_split(idx, test_size=test_split, random_state=42)

    sX = MinMaxScaler(); sY = MinMaxScaler(); sA = MinMaxScaler()
    Xtr = sX.fit_transform(X[ti]); ytr = sY.fit_transform(y[ti].reshape(-1,1)).flatten()
    Xte = sX.transform(X[ei]);     yte = sY.transform(y[ei].reshape(-1,1)).flatten()
    atr = sA.fit_transform(y_aux[ti]); ate = sA.transform(y_aux[ei])

    logging.info(f"Train: {len(Xtr)}, Test: {len(Xte)}")

    data = {
        'train': (torch.tensor(Xtr, dtype=torch.float32),
                  torch.tensor(ytr, dtype=torch.float32),
                  torch.tensor(atr, dtype=torch.float32),
                  torch.tensor(pos[ti], dtype=torch.long)),
        'test':  (torch.tensor(Xte, dtype=torch.float32),
                  torch.tensor(yte, dtype=torch.float32),
                  torch.tensor(ate, dtype=torch.float32),
                  torch.tensor(pos[ei], dtype=torch.long)),
        'scalers': (sX, sY, sA),
        'geo': (ds, tr, mk, gl),
    }
    return data

class ScalerInfo:
    """Scaler info on GPU for physics constraints."""
    def __init__(self, sX, sY, dev='cpu'):
        self.xmin = torch.tensor(sX.data_min_, dtype=torch.float32, device=dev)
        self.xrng = torch.tensor(sX.data_range_, dtype=torch.float32, device=dev)
        self.xrng = torch.where(self.xrng > 1e-10, self.xrng, torch.ones_like(self.xrng))
        self.ymin = torch.tensor(sY.data_min_[0], dtype=torch.float32, device=dev)
        self.yrng = torch.tensor(sY.data_range_[0], dtype=torch.float32, device=dev)
        if self.yrng < 1e-10: self.yrng = torch.tensor(1., dtype=torch.float32, device=dev)
    def inv_ch(self, xn, ch, wa):
        s, e = ch*wa, (ch+1)*wa
        return xn * self.xrng[s:e].unsqueeze(0) + self.xmin[s:e].unsqueeze(0)
    def inv_y(self, yn): return yn * self.yrng + self.ymin
    def to(self, d):
        self.xmin = self.xmin.to(d); self.xrng = self.xrng.to(d)
        self.ymin = self.ymin.to(d); self.yrng = self.yrng.to(d); return self

class DSMulti(Dataset):
    def __init__(self, X, y, aux=None):
        self.X = X; self.y = y; self.aux = aux
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        if self.aux is not None: return self.X[i], self.y[i], self.aux[i]
        return self.X[i], self.y[i], torch.zeros(2)

# ============================================================
# 2. MODEL ZOO
# ============================================================
def _enc(isz, hs):
    l = []; p = isz
    for h in hs: l += [nn.Linear(p, h), nn.ReLU()]; p = h
    return nn.Sequential(*l)

class MLP(nn.Module):
    def __init__(self, isz, hs=[512,256,128], dr=0.3):
        super().__init__(); l = []; p = isz
        for h in hs: l += [nn.Linear(p,h), nn.ReLU(), nn.Dropout(dr)]; p = h
        l.append(nn.Linear(p, 1)); self.m = nn.Sequential(*l)
    def forward(self, x): return self.m(x).squeeze(-1)

class CNN(nn.Module):
    def __init__(self, ws=21, ch=5):
        super().__init__(); self.ws = ws; self.ch = ch
        self.cv = nn.Sequential(nn.Conv2d(ch,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(nn.Linear(32,32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32,1))
    def forward(self, x):
        b = x.shape[0]
        return self.fc(self.cv(x.view(b, self.ch, self.ws, self.ws)).view(b,-1)).squeeze(-1)

class SA(nn.Module):
    def __init__(self, ws=21, hs=[256,128]):
        super().__init__(); wa = ws*ws; hd = hs[-1]; self.wa = wa
        self.encs = nn.ModuleList([_enc(wa, hs) for _ in range(5)])
        self.attn = nn.Sequential(nn.Linear(hd*5, hd), nn.ReLU(), nn.Linear(hd, 5), nn.Softmax(dim=1))
        self.pred = nn.Sequential(nn.Linear(hd, hd//2), nn.ReLU(), nn.Linear(hd//2, 1))
        self.res = nn.Linear(hd*5, hd)
    def forward(self, x):
        wa = self.wa; fs = [self.encs[i](x[:, i*wa:(i+1)*wa]) for i in range(5)]
        c = torch.cat(fs, 1); a = self.attn(c)
        wt = sum(fs[i] * a[:, i:i+1] for i in range(5))
        return self.pred(wt + self.res(c)).squeeze(-1), a

class ESMHSA(nn.Module):
    def __init__(self, ws=21, hs=[256,144], nh=12):
        super().__init__(); wa = ws*ws; hd = hs[-1]; self.wa = wa; self.ws = ws
        self.cnn = nn.Sequential(nn.Conv2d(1,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.Flatten())
        self.encs = nn.ModuleList([_enc(32*wa, hs) for _ in range(5)])
        self.q = nn.Parameter(torch.randn(1, hd))
        self.mha = nn.MultiheadAttention(hd, nh); self.norm = nn.LayerNorm(hd)
        self.pred = nn.Sequential(nn.Linear(hd, hd//2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hd//2, 1))
    def forward(self, x):
        b = x.shape[0]; wa = self.wa; ws = self.ws
        fs = torch.stack([self.encs[i](self.cnn(x[:, i*wa:(i+1)*wa].view(b,1,ws,ws))) for i in range(5)], 0)
        o, aw = self.mha(self.q.expand(1, b, -1), fs, fs)
        return self.pred(self.norm(o.squeeze(0))).squeeze(-1), aw

class FA(nn.Module):
    """Frequency-Aware baseline: FFT + per-channel encoder + attention."""
    def __init__(self, ws=21, hs=[256,128]):
        super().__init__(); wa = ws*ws; hd = hs[-1]; self.wa = wa; self.ws = ws
        self.encs = nn.ModuleList([_enc(wa*2, hs) for _ in range(5)])
        self.attn = nn.Sequential(nn.Linear(hd*5, hd), nn.ReLU(), nn.Linear(hd, 5), nn.Softmax(dim=1))
        self.pred = nn.Sequential(nn.Linear(hd, hd//2), nn.ReLU(), nn.Linear(hd//2, 1))
    def forward(self, x):
        b = x.shape[0]; wa = self.wa; ws = self.ws; enc = []
        for i in range(5):
            w = x[:, i*wa:(i+1)*wa].view(b, ws, ws)
            f = torch.fft.fft2(w).view(b, -1)
            enc.append(self.encs[i](torch.cat([f.real, f.imag], 1)))
        c = torch.cat(enc, 1); a = self.attn(c)
        wt = sum(enc[i] * a[:, i:i+1] for i in range(5))
        return self.pred(wt).squeeze(-1), {'attention_weights': a}

class Proposed(nn.Module):
    """
    Full proposed model: FA + LSF + MSF + CCSI + optional physics modules.
    Flags control ablation variants:
      lsf:  Learnable Spectral Filter (per-frequency gate, identity init)
      msf:  Multi-Scale Spectral Fusion (dual-scale FFT via zero-padding)
      ccsi: Cross-Channel Spectral Interaction (diagonal scale+bias)
      mtl:  Multi-Task Learning (auxiliary GA/VGG prediction heads)
      spi:  Spectral Prior Injection (admittance-based encoder init)
      pfa:  Physics Feature Augmentation (gradient/Laplacian gating)
    """
    def __init__(self, ws=21, hs=[256,128], lsf=True, msf=True, ccsi=True,
                 mtl=False, spi=False, pfa=False, dx=7408.):
        super().__init__()
        wa = ws*ws; hd = hs[-1]
        self.wa = wa; self.ws = ws; self.hd = hd
        self.use_lsf = lsf; self.use_msf = msf; self.use_ccsi = ccsi
        self.use_mtl = mtl; self.use_spi = spi; self.use_pfa = pfa

        # --- LSF ---
        if lsf:
            self.lsf_p = nn.ParameterList([nn.Parameter(torch.zeros(ws, ws)) for _ in range(5)])

        # --- MSF ---
        if msf:
            self.ws_pad = ws * 2
            self.enc_s0 = nn.ModuleList([_enc(ws*ws*2, hs) for _ in range(5)])
            self.enc_s1 = nn.ModuleList([_enc(self.ws_pad**2 * 2, hs) for _ in range(5)])
            self.scale_gate = nn.Sequential(
                nn.Linear(hd*2, hd), nn.ReLU(), nn.Linear(hd, 2), nn.Softmax(dim=1))
        else:
            self.encs = nn.ModuleList([_enc(wa*2, hs) for _ in range(5)])

        # --- SPI ---
        if spi and msf:
            self._apply_spectral_prior(ws, dx)

        # --- CCSI ---
        if ccsi:
            self.ccsi_scale = nn.Parameter(torch.zeros(5))
            self.ccsi_bias = nn.Parameter(torch.zeros(5))

        # --- PFA ---
        if pfa:
            self.pfa_proj = nn.Sequential(nn.Linear(15, hd//4), nn.ReLU())
            self.pfa_gate = nn.Sequential(
                nn.Linear(hd + hd//4, hd), nn.ReLU(), nn.Linear(hd, hd), nn.Sigmoid())

        # --- Shared head ---
        self.attn = nn.Sequential(nn.Linear(hd*5, hd), nn.ReLU(), nn.Linear(hd, 5), nn.Softmax(dim=1))
        self.pred = nn.Sequential(nn.Linear(hd, hd//2), nn.ReLU(), nn.Linear(hd//2, 1))

        # --- MTL ---
        if mtl:
            self.head_ga = nn.Sequential(nn.Linear(hd, hd//4), nn.ReLU(), nn.Linear(hd//4, 1))
            self.head_vgg = nn.Sequential(nn.Linear(hd, hd//4), nn.ReLU(), nn.Linear(hd//4, 1))

    def _apply_spectral_prior(self, ws, dx):
        G = 6.674e-11; rho = 1670.; h0 = 3000.
        fx = torch.fft.fftfreq(ws, d=dx); fy = torch.fft.fftfreq(ws, d=dx)
        fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')
        k = 2 * np.pi * torch.sqrt(fxx**2 + fyy**2)
        Z = 2 * np.pi * G * rho * torch.exp(-k * h0)
        Z = Z / (Z.max() + 1e-20)
        Z_flat = Z.view(-1)
        with torch.no_grad():
            W = self.enc_s0[0][0].weight
            W[:, :ws*ws] *= Z_flat.unsqueeze(0)
            W[:, ws*ws:] *= Z_flat.unsqueeze(0)

    def forward(self, x):
        b = x.shape[0]; wa = self.wa; ws = self.ws
        channel_feats = []; sg_list = []

        # PFA features
        pfa_feats = None
        if self.use_pfa:
            pfa_list = []; c_ = ws // 2
            for ch in range(5):
                w = x[:, ch*wa:(ch+1)*wa].view(b, ws, ws)
                pfa_list.extend([
                    (w[:, c_, c_+1] - w[:, c_, c_-1]) / 2,
                    (w[:, c_+1, c_] - w[:, c_-1, c_]) / 2,
                    w[:, c_+1, c_] + w[:, c_-1, c_] + w[:, c_, c_+1] + w[:, c_, c_-1] - 4*w[:, c_, c_]
                ])
            pfa_feats = self.pfa_proj(torch.stack(pfa_list, dim=1))

        for ch in range(5):
            w = x[:, ch*wa:(ch+1)*wa].view(b, ws, ws)
            freq = torch.fft.fft2(w)

            if self.use_lsf:
                g = 1.0 + 0.3 * torch.tanh(self.lsf_p[ch])
                freq = freq * g.unsqueeze(0)
                sg_list.append(g.detach())

            if self.use_msf:
                fv0 = freq.view(b, -1)
                feat0 = self.enc_s0[ch](torch.cat([fv0.real, fv0.imag], 1))

                wsp = self.ws_pad; pt = wsp - ws; pl = pt//2; pr = pt - pl
                w_pad = F.pad(w, (pl, pr, pl, pr), mode='constant', value=0)
                fv1 = torch.fft.fft2(w_pad).view(b, -1)
                feat1 = self.enc_s1[ch](torch.cat([fv1.real, fv1.imag], 1))

                sw = self.scale_gate(torch.cat([feat0, feat1], 1))
                feat = feat0 * sw[:, 0:1] + feat1 * sw[:, 1:2]
            else:
                fv = freq.view(b, -1)
                feat = self.encs[ch](torch.cat([fv.real, fv.imag], 1))
            channel_feats.append(feat)

        if self.use_ccsi:
            for i in range(5):
                channel_feats[i] = (1.0 + self.ccsi_scale[i]) * channel_feats[i] + self.ccsi_bias[i]

        c_cat = torch.cat(channel_feats, 1)
        a = self.attn(c_cat)
        wt = sum(channel_feats[i] * a[:, i:i+1] for i in range(5))

        if self.use_pfa and pfa_feats is not None:
            gate = self.pfa_gate(torch.cat([wt, pfa_feats], 1))
            wt = wt * gate

        depth = self.pred(wt).squeeze(-1)

        aux = {'attention_weights': a}
        if sg_list: aux['spectral_gates'] = sg_list
        if self.use_ccsi:
            aux['ccsi_scale'] = self.ccsi_scale.detach().cpu().numpy().tolist()
        if self.use_mtl:
            aux['pred_ga'] = self.head_ga(wt).squeeze(-1)
            aux['pred_vgg'] = self.head_vgg(wt).squeeze(-1)
        return depth, aux

# ============================================================
# 3. EXPLICIT PHYSICS CONSTRAINTS
# ============================================================
class MGCLoss(nn.Module):
    def __init__(self, ws=21):
        super().__init__(); self.wa = ws*ws; self.ws = ws; self.c = ws//2
        self.s_e = nn.Parameter(torch.tensor(1.)); self.b_e = nn.Parameter(torch.tensor(0.))
        self.s_n = nn.Parameter(torch.tensor(1.)); self.b_n = nn.Parameter(torch.tensor(0.))
        self.s_v = nn.Parameter(torch.tensor(1.)); self.b_v = nn.Parameter(torch.tensor(0.))
    def forward(self, xn, yp, si):
        b = xn.shape[0]; wa = self.wa; ws = self.ws; c = self.c
        vgg = si.inv_ch(xn[:, wa:wa*2], 1, wa).view(b, ws, ws)
        dn = si.inv_ch(xn[:, wa*2:wa*3], 2, wa).view(b, ws, ws)
        de = si.inv_ch(xn[:, wa*3:wa*4], 3, wa).view(b, ws, ws)
        gb = si.inv_ch(xn[:, wa*4:], 4, wa).view(b, ws, ws)
        d = si.inv_y(yp); co = gb.clone(); co[:, c, c] = d
        dx = (co[:, c, c+1] - co[:, c, c-1]) / 2
        dy = (co[:, c+1, c] - co[:, c-1, c]) / 2
        lap = co[:, c+1, c] + co[:, c-1, c] + co[:, c, c+1] + co[:, c, c-1] - 4*co[:, c, c]
        sx = dx.detach().std().clamp(min=1.)
        sy = dy.detach().std().clamp(min=1.)
        sl = lap.detach().std().clamp(min=1.)
        return (F.mse_loss(dx, self.s_e*de[:, c, c]+self.b_e)/sx**2 +
                F.mse_loss(dy, self.s_n*dn[:, c, c]+self.b_n)/sy**2 +
                F.mse_loss(lap, self.s_v*vgg[:, c, c]+self.b_v)/sl**2)

class PFVLoss(nn.Module):
    def __init__(self, ws=21, dx=7408.):
        super().__init__(); self.wa = ws*ws; self.ws = ws; self.c = ws//2
        self.G = 6.674e-11
        self.lrho = nn.Parameter(torch.tensor(np.log(1670.)))
        self.lh0 = nn.Parameter(torch.tensor(np.log(3000.)))
        fx = torch.fft.fftfreq(ws, d=dx); fy = torch.fft.fftfreq(ws, d=dx)
        fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')
        k = 2*np.pi*torch.sqrt(fxx**2 + fyy**2)
        self.register_buffer('k', k)
        km = (k.view(-1) > 1e-8)
        self.register_buffer('ndc_idx', km.nonzero(as_tuple=False).squeeze(-1))
    def forward(self, xn, yp, si):
        b = xn.shape[0]; wa = self.wa; ws = self.ws; c = self.c
        ga = si.inv_ch(xn[:, :wa], 0, wa).view(b, ws, ws)
        gb = si.inv_ch(xn[:, wa*4:], 4, wa).view(b, ws, ws)
        d = si.inv_y(yp); h = gb.clone(); h[:, c, c] = d
        rho = torch.exp(self.lrho); h0 = torch.exp(self.lh0)
        Df = 2*np.pi*self.G*rho*torch.fft.fft2(h)*torch.exp(-self.k.unsqueeze(0)*h0)
        Do = torch.fft.fft2(ga)
        la_f = torch.log(torch.abs(Df).view(b,-1).index_select(1, self.ndc_idx)+1e-10)
        la_o = torch.log(torch.abs(Do).view(b,-1).index_select(1, self.ndc_idx)+1e-10)
        la_f = (la_f - la_f.mean(1, keepdim=True)) / (la_f.std(1, keepdim=True)+1e-10)
        la_o = (la_o - la_o.mean(1, keepdim=True)) / (la_o.std(1, keepdim=True)+1e-10)
        return (1 - F.cosine_similarity(la_f, la_o, dim=1)).mean()

class SERLoss(nn.Module):
    def __init__(self, ws=21):
        super().__init__(); self.wa = ws*ws; self.ws = ws; self.c = ws//2
        self.beta = nn.Parameter(torch.tensor(3.0))
        fx = torch.fft.fftfreq(ws); fy = torch.fft.fftfreq(ws)
        fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')
        r = torch.sqrt(fxx**2 + fyy**2).view(-1)
        vm = r > 0.05
        self.register_buffer('log_r', torch.log(r[vm]+1e-10))
        self.register_buffer('vm_idx', vm.nonzero(as_tuple=False).squeeze(-1))
    def forward(self, xn, yp, si):
        b = xn.shape[0]; wa = self.wa; ws = self.ws; c = self.c
        gb = si.inv_ch(xn[:, wa*4:], 4, wa).view(b, ws, ws)
        d = si.inv_y(yp); co = gb.clone(); co[:, c, c] = d
        sp = torch.fft.fft2(co); pw = (sp.real**2 + sp.imag**2).view(b, -1)
        lp = torch.log(pw.index_select(1, self.vm_idx).mean(0)+1e-10)
        lf = self.log_r; fc = lf - lf.mean(); pc = lp - lp.mean()
        dn = (fc**2).sum()
        if dn > 1e-10: slope = (fc*pc).sum()/dn
        else: return torch.tensor(0., device=xn.device, requires_grad=True)
        return F.mse_loss(slope, -torch.abs(self.beta))

# ============================================================
# 4. TRAINING
# ============================================================
def train_model(model, Xtr, ytr, aux_tr=None, epochs=300, val_split=0.2, bs=64,
                cons=None, si=None, c_start=100, c_warm=50, c_alpha=0.2,
                mtl_weight=0.3):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    cons = cons or {}; hp = len(cons) > 0
    for n in cons: cons[n] = cons[n].to(dev)
    if si: si = si.to(dev)

    has_mtl = hasattr(model, 'use_mtl') and model.use_mtl
    ds = DSMulti(Xtr, ytr, aux_tr if has_mtl else None)
    vsz = int(len(ds)*val_split); tsz = len(ds) - vsz
    tds, vds = random_split(ds, [tsz, vsz])
    tdl = DataLoader(tds, batch_size=bs, shuffle=True)
    vdl = DataLoader(vds, batch_size=bs, shuffle=False)
    logging.info(f"Train: {tsz}, Val: {vsz}")

    allp = list(model.parameters())
    for c in cons.values(): allp += list(c.parameters())
    crit = nn.MSELoss()
    opt = torch.optim.Adam(allp, lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=10)

    history = {'train_loss': [], 'val_loss': [], 'mtl_loss': [], 'cons_loss': {n: [] for n in cons}}
    best_vl = float('inf'); best_st = None; pat = 50; pat_c = 0; ema_mse = None

    for ep in tqdm(range(epochs), desc="Training"):
        if ep < 10:
            for pg in opt.param_groups: pg['lr'] = 1e-5 + 9e-5 * ep / 10
        elif ep == 10:
            for pg in opt.param_groups: pg['lr'] = 1e-4

        cs = 0. if ep < c_start else min(1., (ep-c_start)/c_warm) if ep < c_start+c_warm else 1.
        if ep == c_start and hp:
            pat_c = 0; logging.info(f"Constraints ON at ep {ep}. Best val: {best_vl:.6f}")

        model.train()
        for c in cons.values(): c.train()
        tmse = 0.; tmtl = 0.; ecl = {n: 0. for n in cons}; ns = 0

        for batch in tdl:
            Xb, yb = batch[0].to(dev), batch[1].to(dev)
            ab = batch[2].to(dev) if has_mtl else None
            opt.zero_grad()

            out = model(Xb)
            yp = out[0] if isinstance(out, tuple) else out
            aux = out[1] if isinstance(out, tuple) else {}
            if not isinstance(aux, dict): aux = {'attention_weights': aux}

            mse_loss = crit(yp, yb); loss = mse_loss.clone()

            # Entropy regularization
            at = aux.get('attention_weights')
            if at is not None and isinstance(at, torch.Tensor) and at.dim() >= 2:
                loss = loss - 0.01 * (-torch.sum(at * torch.log(at + 1e-10), dim=-1).mean())

            # MTL
            mtl_val = 0.
            if has_mtl and 'pred_ga' in aux and ab is not None:
                l_ga = crit(aux['pred_ga'], ab[:, 0])
                l_vgg = crit(aux['pred_vgg'], ab[:, 1])
                mtl_loss = l_ga + l_vgg
                loss = loss + mtl_weight * mtl_loss
                mtl_val = mtl_loss.item()

            # Explicit physics constraints
            if cs > 0 and hp and si is not None:
                mv = mse_loss.detach().item()
                if ema_mse is None: ema_mse = mv
                else: ema_mse = 0.9 * ema_mse + 0.1 * mv
                for n, con in cons.items():
                    cl = con(Xb, yp, si); cv = cl.detach().item()
                    aw = min(c_alpha * ema_mse / cv, 0.1) if cv > 1e-10 else 0.
                    loss = loss + cs * aw * cl
                    ecl[n] += cv * len(Xb)

            loss.backward(); torch.nn.utils.clip_grad_norm_(allp, 5.0); opt.step()
            tmse += mse_loss.item() * len(Xb); tmtl += mtl_val * len(Xb); ns += len(Xb)

        tl = tmse / max(ns, 1)
        history['train_loss'].append(tl)
        history['mtl_loss'].append(tmtl / max(ns, 1))
        for n in cons: history['cons_loss'][n].append(ecl[n] / max(ns, 1))

        # Validation
        model.eval(); vtot = 0; nv = 0
        with torch.no_grad():
            for batch in vdl:
                Xb, yb = batch[0].to(dev), batch[1].to(dev)
                out = model(Xb); yp = out[0] if isinstance(out, tuple) else out
                vtot += crit(yp, yb).item() * len(Xb); nv += len(Xb)
        vl = vtot / max(nv, 1); history['val_loss'].append(vl); sched.step(vl)

        if vl < best_vl:
            best_vl = vl; best_st = {k: v.clone() for k, v in model.state_dict().items()}; pat_c = 0
        else:
            pat_c += 1

        if (ep+1) % 50 == 0 or ep == 0:
            msg = f"Ep {ep+1}/{epochs}, Tr: {tl:.6f}, Vl: {vl:.6f}, Cs: {cs:.2f}"
            if tmtl > 0: msg += f", MTL: {history['mtl_loss'][-1]:.6f}"
            for n in cons: msg += f", {n}: {history['cons_loss'][n][-1]:.4f}"
            if ema_mse and hp: msg += f", ema: {ema_mse:.6f}"
            logging.info(msg)

        if pat_c >= pat:
            logging.info(f"Early stop at ep {ep+1}"); break

    if best_st: model.load_state_dict(best_st)
    return model, history

# ============================================================
# 5. EVALUATE
# ============================================================
def evaluate(model, Xte, yte, scY, pte, ds, tr, mk, gl, name, ws=21, bs=64, region="r"):
    """Evaluate and return metrics + full prediction arrays for visualization."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.eval()
    dl = DataLoader(DSMulti(Xte, yte), batch_size=bs, shuffle=False)
    yps = []; atts = []
    with torch.no_grad():
        for batch in dl:
            Xb = batch[0].to(dev)
            out = model(Xb)
            yp = out[0] if isinstance(out, tuple) else out
            yps.append(yp.cpu().numpy())
            aux = out[1] if isinstance(out, tuple) else {}
            if isinstance(aux, dict) and 'attention_weights' in aux:
                atts.append(aux['attention_weights'].cpu().numpy())

    yp_np = np.concatenate(yps)
    yp_orig = scY.inverse_transform(yp_np.reshape(-1, 1)).flatten()
    yt_orig = scY.inverse_transform(yte.cpu().numpy().reshape(-1, 1)).flatten()
    pos_np = pte.cpu().numpy()

    vm = mk[pos_np[:, 0], pos_np[:, 1]]
    if not np.all(vm):
        yp_orig = yp_orig[vm]; yt_orig = yt_orig[vm]; pos_np = pos_np[vm]

    rmse = np.sqrt(mean_squared_error(yt_orig, yp_orig))
    mae = mean_absolute_error(yt_orig, yp_orig)

    # GEBCO reference
    gt = np.array([gl[r, c] for r, c in pos_np])
    grmse = np.sqrt(mean_squared_error(yt_orig, gt)) if len(gt) > 0 else np.nan
    improv = (grmse - rmse) / grmse * 100 if not np.isnan(grmse) else np.nan

    # Error grid for spatial visualization
    err_grid = np.full(ds, np.nan)
    pred_grid = np.full(ds, np.nan)
    for idx, (r, c) in enumerate(pos_np):
        err_grid[r, c] = yt_orig[idx] - yp_orig[idx]
        pred_grid[r, c] = yp_orig[idx]

    logging.info(f"===== {name} [{region}] ===== RMSE: {rmse:.2f} | MAE: {mae:.2f} | Improv: {improv:.1f}%")

    return {
        'model': name, 'region': region, 'rmse': float(rmse), 'mae': float(mae),
        'grmse': float(grmse), 'improv': float(improv),
        # Arrays for plotting (saved separately)
        '_pred': yp_orig, '_true': yt_orig, '_pos': pos_np,
        '_err_grid': err_grid, '_pred_grid': pred_grid,
        '_attn': np.concatenate(atts, 0) if atts else None,
    }

# ============================================================
# 6. EXPERIMENT RUNNER
# ============================================================
def run_experiment_group(name, model_configs, data, args, outdir, sinfo=None):
    """Run a group of models across all seeds, save everything."""
    Xtr, ytr, atr, _ = data['train']
    Xte, yte, ate, pte = data['test']
    _, scY, _ = data['scalers']
    ds, tr, mk, gl = data['geo']

    all_results = []
    all_arrays = {}

    for seed in args.seeds:
        logging.info(f"\n{'#'*60}\n# {name} - SEED={seed}\n{'#'*60}")
        for mn, cfg in model_configs.items():
            set_seed(seed)
            model = cfg['model_fn']()
            cons = cfg.get('cons_fn', lambda: {})()
            hp = len(cons) > 0

            logging.info(f"\n--- {mn} (seed={seed}) ---")
            model, hist = train_model(
                model, Xtr, ytr, aux_tr=atr,
                epochs=args.epochs, val_split=0.2, bs=args.bs,
                cons=cons, si=sinfo if hp else None,
                c_start=100, c_warm=50, c_alpha=0.2, mtl_weight=0.3
            )

            met = evaluate(model, Xte, yte, scY, pte, ds, tr, mk, gl,
                           mn, args.ws, args.bs, args.region)
            met['seed'] = seed
            met['group'] = name

            # Extract arrays, save separately
            key = f"{mn}_seed{seed}"
            all_arrays[key] = {
                'pred': met.pop('_pred'), 'true': met.pop('_true'),
                'pos': met.pop('_pos'), 'err_grid': met.pop('_err_grid'),
                'pred_grid': met.pop('_pred_grid'), 'attn': met.pop('_attn'),
                'train_loss': hist['train_loss'], 'val_loss': hist['val_loss'],
                'mtl_loss': hist['mtl_loss'],
                'cons_loss': hist['cons_loss'],
            }
            all_results.append(met)

    # Save
    grp_dir = os.path.join(outdir, name)
    os.makedirs(grp_dir, exist_ok=True)

    # Metrics CSV
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(grp_dir, 'metrics.csv'), index=False)

    # Arrays NPZ (for plotting)
    np.savez_compressed(os.path.join(grp_dir, 'arrays.npz'), **{
        k: v for key_dict in all_arrays.values() for k, v in
        {f"{outer_k}_{inner_k}": inner_v for outer_k, inner_v_dict in [(kk, all_arrays[kk]) for kk in all_arrays]
         for inner_k, inner_v in inner_v_dict.items() if inner_v is not None and not isinstance(inner_v, dict)}.items()
    })
    # Save arrays as pickle (more flexible for nested dicts)
    with open(os.path.join(grp_dir, 'arrays.pkl'), 'wb') as f:
        pickle.dump(all_arrays, f)

    # Summary
    model_names = list(model_configs.keys())
    logging.info(f"\n{'='*60}\n{name} SUMMARY\n{'='*60}")
    agg = []
    for mn in model_names:
        rs = [r['rmse'] for r in all_results if r['model'] == mn]
        ms = [r['mae'] for r in all_results if r['model'] == mn]
        ips = [r['improv'] for r in all_results if r['model'] == mn]
        agg.append({
            'Model': mn,
            'RMSE': f"{np.mean(rs):.2f}±{np.std(rs):.2f}",
            'MAE': f"{np.mean(ms):.2f}±{np.std(ms):.2f}",
            'Improv%': f"{np.mean(ips):.1f}±{np.std(ips):.1f}",
        })
    df_agg = pd.DataFrame(agg)
    logging.info(f"\n{df_agg.to_string(index=False)}")
    df_agg.to_csv(os.path.join(grp_dir, 'summary.csv'), index=False)

    return all_results

# ============================================================
# 7. MAIN
# ============================================================
def main():
    args = parse_args()
    outdir = os.path.join(args.outdir, args.region)
    os.makedirs(outdir, exist_ok=True)
    logging.info(f"Region={args.region}, WS={args.ws}, Seeds={args.seeds}, Output={outdir}")

    # Load data
    data = load_data(args.region, ws=args.ws)
    sX, sY, sA = data['scalers']
    sinfo = ScalerInfo(sX, sY)

    WS = args.ws
    all_results = []

    # ==================== A: Architecture Ablation ====================
    if not args.skip_arch:
        arch_cfgs = {
            "MLP":          {'model_fn': lambda: MLP(WS*WS*5)},
            "CNN":          {'model_fn': lambda: CNN(WS)},
            "SA":           {'model_fn': lambda: SA(WS)},
            "ES-MHSA":     {'model_fn': lambda: ESMHSA(WS, [256,144], 12)},
            "FA":           {'model_fn': lambda: FA(WS)},
            "FA+LSF":       {'model_fn': lambda: Proposed(WS, lsf=True, msf=False, ccsi=False)},
            "FA+LSF+CCSI":  {'model_fn': lambda: Proposed(WS, lsf=True, msf=False, ccsi=True)},
            "FA+MSF":       {'model_fn': lambda: Proposed(WS, lsf=True, msf=True, ccsi=False)},
            "FA+MSF+CCSI":  {'model_fn': lambda: Proposed(WS, lsf=True, msf=True, ccsi=True)},
        }
        r = run_experiment_group("arch_ablation", arch_cfgs, data, args, outdir)
        all_results.extend(r)

    # ==================== B: Explicit Physics Constraints ====================
    if not args.skip_phys:
        phys_cfgs = {
            "Proposed":      {'model_fn': lambda: Proposed(WS)},
            "Proposed+MGC":  {'model_fn': lambda: Proposed(WS),
                              'cons_fn': lambda: {'mgc': MGCLoss(WS)}},
            "Proposed+PFV":  {'model_fn': lambda: Proposed(WS),
                              'cons_fn': lambda: {'pfv': PFVLoss(WS)}},
            "Proposed+SER":  {'model_fn': lambda: Proposed(WS),
                              'cons_fn': lambda: {'ser': SERLoss(WS)}},
            "Proposed+All":  {'model_fn': lambda: Proposed(WS),
                              'cons_fn': lambda: {'mgc': MGCLoss(WS), 'pfv': PFVLoss(WS), 'ser': SERLoss(WS)}},
        }
        r = run_experiment_group("explicit_physics", phys_cfgs, data, args, outdir, sinfo)
        all_results.extend(r)

    # ==================== C: Implicit Physics ====================
    if not args.skip_implicit:
        impl_cfgs = {
            "Proposed":          {'model_fn': lambda: Proposed(WS)},
            "Proposed+MTL":      {'model_fn': lambda: Proposed(WS, mtl=True)},
            "Proposed+SPI":      {'model_fn': lambda: Proposed(WS, spi=True)},
            "Proposed+PFA":      {'model_fn': lambda: Proposed(WS, pfa=True)},
            "Proposed+MTL+SPI":  {'model_fn': lambda: Proposed(WS, mtl=True, spi=True)},
            "Proposed+MTL+PFA":  {'model_fn': lambda: Proposed(WS, mtl=True, pfa=True)},
            "Proposed+ALL":      {'model_fn': lambda: Proposed(WS, mtl=True, spi=True, pfa=True)},
        }
        r = run_experiment_group("implicit_physics", impl_cfgs, data, args, outdir, sinfo)
        all_results.extend(r)

    # ==================== Save global results ====================
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(outdir, 'all_results.csv'), index=False)
    with open(os.path.join(outdir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # ==================== Save config ====================
    config = {
        'region': args.region, 'ws': args.ws, 'seeds': args.seeds,
        'epochs': args.epochs, 'bs': args.bs,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(outdir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    logging.info(f"\n{'='*60}\nALL DONE: {args.region} ws={args.ws}\n{'='*60}")
    logging.info(f"Results saved to: {outdir}")

if __name__ == "__main__":
    main()
