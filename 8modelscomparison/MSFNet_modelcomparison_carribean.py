"""
MSFNet_paper_final.py — Final Paper Experiment: 8 Models × 4 Regions × 3 Seeds

Outputs ALL data needed for paper tables and figures:
  - metrics.csv: RMSE/MAE/Improv per model per seed
  - arrays.pkl: pred/true/pos/err_grid/pred_grid/attn/train_loss/val_loss
  - summary.csv: aggregated statistics
  - gebco_grid.npy: GEBCO low-res grid for reference
  - truth_grid.npy: truth grid for PSD analysis
  - mask.npy: valid pixel mask

Models (8): MLP, CNN, Transformer, SA, ES-MHSA, TransUNet, FA, Proposed
Seeds: [42, 2024, 577]
Regions: xisha, carribean, mexico, pac

★★★ MODIFY REGION CONFIG BELOW FOR EACH REGION ★★★

Usage:
  # Run 4 regions in parallel:
  # Copy this file 4 times, change REGION config, then:
  # python MSFNet_paper_final_xisha.py &
  # python MSFNet_paper_final_carribean.py &
  # python MSFNet_paper_final_mexico.py &
  # python MSFNet_paper_final_pac.py &
  # wait
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os, json, time, pickle
from tqdm import tqdm
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ============================================================
# ★★★ REGION CONFIG — MODIFY HERE ★★★
# ============================================================
REGION = "carribean"
GA_FILE   = "../SWOTcarribean.npy"
VGG_FILE  = "../SWOTVGGcarribean.npy"
SSH_FILE  = "../SWOTDOVnorthcarribean.npy"
DOV_FILE  = "../SWOTDOVeastcarribean.npy"
GEBCO_FILE = "../b_carribean.npy"
TRUTH_FILE = "../carribeantrue2.npy"

# ============================================================
# EXPERIMENT CONFIG (DO NOT CHANGE)
# ============================================================
WS     = 21
SEEDS  = [42, 2024, 577]
N_EP   = 300
BS     = 64
OUTDIR = f"results_paper_{REGION}"

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ============================================================
# 1. DATA
# ============================================================
def _w(d, r, c, ws):
    h = ws // 2; p = np.pad(d, h, mode='mean'); return p[r:r+ws, c:c+ws]

def load_data():
    logging.info(f"Loading data for region={REGION}, ws={WS}")
    ga = np.load(GA_FILE).squeeze(); vgg = np.load(VGG_FILE).squeeze()
    ssh = np.load(SSH_FILE).squeeze(); dov = np.load(DOV_FILE).squeeze()
    geb = np.load(GEBCO_FILE); mb = np.load(TRUTH_FILE)
    assert ga.shape == vgg.shape == ssh.shape == dov.shape
    ds = ga.shape; logging.info(f"Shape: {ds}")
    for arr in [ga, vgg, ssh, dov, geb]:
        arr[np.isnan(arr)] = np.nanmean(arr)
    tr = mb[::4, ::4]; gl = geb[::4, ::4]; mk = ~np.isnan(tr)
    assert mk.shape == ds; logging.info(f"Valid pixels: {mk.sum()}")
    X, y, pos = [], [], []
    for i in range(ds[0]):
        for j in range(ds[1]):
            if mk[i, j]:
                feat = np.concatenate([_w(ga,i,j,WS).flatten(), _w(vgg,i,j,WS).flatten(),
                    _w(ssh,i,j,WS).flatten(), _w(dov,i,j,WS).flatten(), _w(gl,i,j,WS).flatten()])
                X.append(feat); y.append(tr[i,j]); pos.append([i,j])
    X, y, pos = np.array(X), np.array(y), np.array(pos)
    logging.info(f"Samples: {len(X)}, features: {X.shape[1]}")
    idx = np.arange(len(X))
    ti, ei = train_test_split(idx, test_size=0.2, random_state=42)
    sX = MinMaxScaler(); sY = MinMaxScaler()
    Xtr = sX.fit_transform(X[ti]); ytr = sY.fit_transform(y[ti].reshape(-1,1)).flatten()
    Xte = sX.transform(X[ei]);     yte = sY.transform(y[ei].reshape(-1,1)).flatten()
    logging.info(f"Train: {len(Xtr)}, Test: {len(Xte)}")
    return (torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32),
            torch.tensor(pos[ti], dtype=torch.long)), \
           (torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.float32),
            torch.tensor(pos[ei], dtype=torch.long)), \
           (sX, sY), (ds, tr, mk, gl)

class DS(Dataset):
    def __init__(self, X, y): self.X = X; self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ============================================================
# 2. MODELS
# ============================================================
def _enc(isz, hs):
    l = []; p = isz
    for h in hs: l += [nn.Linear(p, h), nn.ReLU()]; p = h
    return nn.Sequential(*l)

# ---------- MLP ----------
class MLP(nn.Module):
    def __init__(self, isz, hs=[512,256,128], dr=0.3):
        super().__init__(); l = []; p = isz
        for h in hs: l += [nn.Linear(p,h), nn.ReLU(), nn.Dropout(dr)]; p = h
        l.append(nn.Linear(p, 1)); self.m = nn.Sequential(*l)
    def forward(self, x): return self.m(x).squeeze(-1)

# ---------- CNN (3-layer version matching original 10-seed experiments) ----------
class CNN(nn.Module):
    def __init__(self, ws=21, ch=5):
        super().__init__(); self.ws = ws; self.ch = ch
        self.cv = nn.Sequential(
            nn.Conv2d(ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        b = x.shape[0]
        return self.fc(self.cv(x.view(b, self.ch, self.ws, self.ws)).view(b, -1)).squeeze(-1)

# ---------- Transformer ----------
class Transformer(nn.Module):
    def __init__(self, ws=21, hs=[256,128], nhead=8, nlayers=3, dr=0.1):
        super().__init__(); wa = ws*ws; hd = hs[-1]; self.wa = wa
        self.embed = nn.ModuleList([_enc(wa, hs) for _ in range(5)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, hd))
        enc_layer = nn.TransformerEncoderLayer(d_model=hd, nhead=nhead,
            dim_feedforward=hd*4, dropout=dr, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.pred = nn.Sequential(nn.Linear(hd, hd//2), nn.ReLU(), nn.Linear(hd//2, 1))
    def forward(self, x):
        b = x.shape[0]; wa = self.wa
        tokens = torch.cat([self.embed[i](x[:, i*wa:(i+1)*wa]).unsqueeze(1) for i in range(5)], dim=1)
        seq = torch.cat([self.cls_token.expand(b, -1, -1), tokens], dim=1)
        return self.pred(self.transformer(seq)[:, 0, :]).squeeze(-1)

# ---------- SA ----------
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

# ---------- ES-MHSA ----------
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

# ---------- TransUNet ----------
class TransUNet(nn.Module):
    def __init__(self, ws=21, ch=5, embed_dim=128, nhead=4, nlayers=2):
        super().__init__(); self.ws = ws; self.ch = ch
        self.enc1 = nn.Sequential(
            nn.Conv2d(ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.pool2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        h1 = (ws + 1) // 2; h2 = (h1 + 1) // 2; self.h2 = h2
        self.patch_embed = nn.Linear(128, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, h2*h2, embed_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
            dim_feedforward=embed_dim*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.proj_back = nn.Linear(embed_dim, 128)
        self.up1 = nn.Sequential(nn.Upsample(size=h1), nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up2 = nn.Sequential(nn.Upsample(size=ws), nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        b = x.shape[0]; x2d = x.view(b, self.ch, self.ws, self.ws)
        e1 = self.enc1(x2d); e1d = self.pool1(e1); e2 = self.enc2(e1d); e2d = self.pool2(e2)
        h2 = self.h2; tokens = e2d.view(b, 128, h2*h2).permute(0, 2, 1)
        tokens = self.patch_embed(tokens) + self.pos_embed
        tokens = self.transformer(tokens)
        feat = self.proj_back(tokens).permute(0, 2, 1).view(b, 128, h2, h2)
        return self.head(self.up2(self.up1(feat))).squeeze(-1)

# ---------- FA ----------
class FA(nn.Module):
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

# ---------- Proposed (MSF-Net) ----------
class Proposed(nn.Module):
    def __init__(self, ws=21, hs=[256,128]):
        super().__init__()
        wa = ws*ws; hd = hs[-1]
        self.wa = wa; self.ws = ws; self.hd = hd
        self.lsf_p = nn.ParameterList([nn.Parameter(torch.zeros(ws, ws)) for _ in range(5)])
        self.ws_pad = ws * 2
        self.enc_s0 = nn.ModuleList([_enc(ws*ws*2, hs) for _ in range(5)])
        self.enc_s1 = nn.ModuleList([_enc(self.ws_pad**2 * 2, hs) for _ in range(5)])
        self.scale_gate = nn.Sequential(
            nn.Linear(hd*2, hd), nn.ReLU(), nn.Linear(hd, 2), nn.Softmax(dim=1))
        self.ccsi_scale = nn.Parameter(torch.zeros(5))
        self.ccsi_bias = nn.Parameter(torch.zeros(5))
        self.attn = nn.Sequential(nn.Linear(hd*5, hd), nn.ReLU(), nn.Linear(hd, 5), nn.Softmax(dim=1))
        self.pred = nn.Sequential(nn.Linear(hd, hd//2), nn.ReLU(), nn.Linear(hd//2, 1))
    def forward(self, x):
        b = x.shape[0]; wa = self.wa; ws = self.ws
        channel_feats = []; sg_list = []
        for ch in range(5):
            w = x[:, ch*wa:(ch+1)*wa].view(b, ws, ws)
            freq = torch.fft.fft2(w)
            g = 1.0 + 0.3 * torch.tanh(self.lsf_p[ch])
            freq = freq * g.unsqueeze(0)
            sg_list.append(g.detach())
            fv0 = freq.view(b, -1)
            feat0 = self.enc_s0[ch](torch.cat([fv0.real, fv0.imag], 1))
            wsp = self.ws_pad; pt = wsp - ws; pl = pt//2; pr = pt - pl
            w_pad = F.pad(w, (pl, pr, pl, pr), mode='constant', value=0)
            fv1 = torch.fft.fft2(w_pad).view(b, -1)
            feat1 = self.enc_s1[ch](torch.cat([fv1.real, fv1.imag], 1))
            sw = self.scale_gate(torch.cat([feat0, feat1], 1))
            feat = feat0 * sw[:, 0:1] + feat1 * sw[:, 1:2]
            channel_feats.append(feat)
        for i in range(5):
            channel_feats[i] = (1.0 + self.ccsi_scale[i]) * channel_feats[i] + self.ccsi_bias[i]
        c_cat = torch.cat(channel_feats, 1)
        a = self.attn(c_cat)
        wt = sum(channel_feats[i] * a[:, i:i+1] for i in range(5))
        depth = self.pred(wt).squeeze(-1)
        return depth, {'attention_weights': a, 'spectral_gates': sg_list}

# ============================================================
# 3. TRAINING (identical to newaccxisha.py)
# ============================================================
def train_model(model, Xtr, ytr, epochs=300, val_split=0.2, bs=64):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    ds = DS(Xtr, ytr)
    vsz = int(len(ds)*val_split); tsz = len(ds) - vsz
    tds, vds = random_split(ds, [tsz, vsz])
    tdl = DataLoader(tds, batch_size=bs, shuffle=True)
    vdl = DataLoader(vds, batch_size=bs, shuffle=False)
    logging.info(f"Train: {tsz}, Val: {vsz}")
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=10)
    history = {'train_loss': [], 'val_loss': []}
    best_vl = float('inf'); best_st = None; pat = 50; pat_c = 0

    for ep in tqdm(range(epochs), desc="Training"):
        if ep < 10:
            for pg in opt.param_groups: pg['lr'] = 1e-5 + 9e-5 * ep / 10
        elif ep == 10:
            for pg in opt.param_groups: pg['lr'] = 1e-4
        model.train(); tmse = 0.; ns = 0
        for Xb, yb in tdl:
            Xb, yb = Xb.to(dev), yb.to(dev); opt.zero_grad()
            out = model(Xb)
            yp = out[0] if isinstance(out, tuple) else out
            aux = out[1] if isinstance(out, tuple) else {}
            if not isinstance(aux, dict): aux = {'attention_weights': aux}
            mse_loss = crit(yp, yb); loss = mse_loss.clone()
            at = aux.get('attention_weights')
            if at is not None and isinstance(at, torch.Tensor) and at.dim() >= 2:
                loss = loss - 0.01 * (-torch.sum(at * torch.log(at + 1e-10), dim=-1).mean())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tmse += mse_loss.item() * len(Xb); ns += len(Xb)
        tl = tmse / max(ns, 1); history['train_loss'].append(tl)
        model.eval(); vtot = 0; nv = 0
        with torch.no_grad():
            for Xb, yb in vdl:
                Xb, yb = Xb.to(dev), yb.to(dev)
                out = model(Xb); yp = out[0] if isinstance(out, tuple) else out
                vtot += crit(yp, yb).item() * len(Xb); nv += len(Xb)
        vl = vtot / max(nv, 1); history['val_loss'].append(vl); sched.step(vl)
        if vl < best_vl:
            best_vl = vl; best_st = {k: v.clone() for k, v in model.state_dict().items()}; pat_c = 0
        else:
            pat_c += 1
        if (ep+1) % 50 == 0 or ep == 0:
            logging.info(f"Ep {ep+1}/{epochs}, Tr: {tl:.6f}, Vl: {vl:.6f}")
        if pat_c >= pat:
            logging.info(f"Early stop at ep {ep+1}"); break
    if best_st: model.load_state_dict(best_st)
    return model, history

# ============================================================
# 4. EVALUATE (saves all arrays for figure generation)
# ============================================================
def evaluate(model, Xte, yte, scY, pte, ds, tr, mk, gl, name, ws=21, bs=64):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.eval()
    dl = DataLoader(DS(Xte, yte), batch_size=bs, shuffle=False)
    yps = []; atts = []; sgs = []
    with torch.no_grad():
        for Xb, _ in dl:
            Xb = Xb.to(dev)
            out = model(Xb); yp = out[0] if isinstance(out, tuple) else out
            yps.append(yp.cpu().numpy())
            aux = out[1] if isinstance(out, tuple) else {}
            if isinstance(aux, dict):
                if 'attention_weights' in aux:
                    aw = aux['attention_weights']
                    if isinstance(aw, torch.Tensor): atts.append(aw.cpu().numpy())
                if 'spectral_gates' in aux:
                    sgs.append([g.cpu().numpy() for g in aux['spectral_gates']])
    yp_np = np.concatenate(yps)
    yp_orig = scY.inverse_transform(yp_np.reshape(-1, 1)).flatten()
    yt_orig = scY.inverse_transform(yte.cpu().numpy().reshape(-1, 1)).flatten()
    pos_np = pte.cpu().numpy()
    vm = mk[pos_np[:, 0], pos_np[:, 1]]
    if not np.all(vm):
        yp_orig = yp_orig[vm]; yt_orig = yt_orig[vm]; pos_np = pos_np[vm]
    rmse = np.sqrt(mean_squared_error(yt_orig, yp_orig))
    mae = mean_absolute_error(yt_orig, yp_orig)
    gt = np.array([gl[r, c] for r, c in pos_np])
    grmse = np.sqrt(mean_squared_error(yt_orig, gt)) if len(gt) > 0 else np.nan
    improv = (grmse - rmse) / grmse * 100 if not np.isnan(grmse) else np.nan
    # Build full grids for spatial analysis and PSD
    err_grid = np.full(ds, np.nan); pred_grid = np.full(ds, np.nan); true_grid = np.full(ds, np.nan)
    for idx, (r, c) in enumerate(pos_np):
        err_grid[r, c] = yt_orig[idx] - yp_orig[idx]
        pred_grid[r, c] = yp_orig[idx]
        true_grid[r, c] = yt_orig[idx]
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"===== {name} [{REGION}] ===== RMSE: {rmse:.2f} | MAE: {mae:.2f} | Improv: {improv:.1f}% | Params: {nparams:,}")
    return {
        'model': name, 'region': REGION, 'rmse': float(rmse), 'mae': float(mae),
        'grmse': float(grmse), 'improv': float(improv), 'params': nparams,
        '_pred': yp_orig, '_true': yt_orig, '_pos': pos_np,
        '_err_grid': err_grid, '_pred_grid': pred_grid, '_true_grid': true_grid,
        '_attn': np.concatenate(atts, 0) if atts else None,
        '_spectral_gates': sgs[0] if sgs else None,  # from last batch
    }

# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    logging.info(f"Region={REGION}, WS={WS}, Seeds={SEEDS}, Output={OUTDIR}")

    np.random.seed(42)
    (Xtr, ytr, ptr), (Xte, yte, pte), (sX, sY), (dshape, truth, mask, gebco_lr) = load_data()

    # Save reference grids for figure generation
    np.save(os.path.join(OUTDIR, 'gebco_grid.npy'), gebco_lr)
    np.save(os.path.join(OUTDIR, 'truth_grid.npy'), truth)
    np.save(os.path.join(OUTDIR, 'mask.npy'), mask)
    np.save(os.path.join(OUTDIR, 'dshape.npy'), np.array(dshape))
    logging.info(f"Saved reference grids to {OUTDIR}")

    # ========== Model Configurations ==========
    def get_models():
        return {
            "MLP":          lambda: MLP(WS*WS*5),
            "CNN":          lambda: CNN(WS),
            "Transformer":  lambda: Transformer(WS),
            "SA":           lambda: SA(WS),
            "ES-MHSA":     lambda: ESMHSA(WS, [256,144], 12),
            "TransUNet":    lambda: TransUNet(WS),
            "FA":           lambda: FA(WS),
            "Proposed":     lambda: Proposed(WS),
        }

    all_results = []
    all_arrays = {}

    for seed in SEEDS:
        logging.info(f"\n{'#'*60}\n# ARCH COMPARISON - SEED={seed}\n{'#'*60}")
        for mn, mfn in get_models().items():
            set_seed(seed)
            model = mfn()
            nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f"\n--- {mn} (seed={seed}, params={nparams:,}) ---")

            model, hist = train_model(model, Xtr, ytr, N_EP, 0.2, BS)
            met = evaluate(model, Xte, yte, sY, pte, dshape, truth, mask, gebco_lr, mn, WS, BS)
            met['seed'] = seed

            key = f"{mn}_seed{seed}"
            all_arrays[key] = {
                'pred': met.pop('_pred'), 'true': met.pop('_true'),
                'pos': met.pop('_pos'), 'err_grid': met.pop('_err_grid'),
                'pred_grid': met.pop('_pred_grid'), 'true_grid': met.pop('_true_grid'),
                'attn': met.pop('_attn'),
                'spectral_gates': met.pop('_spectral_gates'),
                'train_loss': hist['train_loss'], 'val_loss': hist['val_loss'],
            }
            all_results.append(met)

    # ========== Save ==========
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUTDIR, 'metrics.csv'), index=False)

    with open(os.path.join(OUTDIR, 'arrays.pkl'), 'wb') as f:
        pickle.dump(all_arrays, f)

    # Summary
    logging.info(f"\n{'='*70}\nARCH COMPARISON SUMMARY ({REGION}, ws={WS})\n{'='*70}")
    model_names = list(get_models().keys())
    agg = []
    for mn in model_names:
        rs = [r['rmse'] for r in all_results if r['model'] == mn]
        ms = [r['mae'] for r in all_results if r['model'] == mn]
        ips = [r['improv'] for r in all_results if r['model'] == mn]
        ps = [r['params'] for r in all_results if r['model'] == mn]
        agg.append({
            'Model': mn, 'Params': f"{ps[0]:,}",
            'RMSE': f"{np.mean(rs):.2f}±{np.std(rs):.2f}",
            'MAE': f"{np.mean(ms):.2f}±{np.std(ms):.2f}",
            'Improv%': f"{np.mean(ips):.1f}±{np.std(ips):.1f}",
        })
    df_agg = pd.DataFrame(agg)
    logging.info(f"\n{df_agg.to_string(index=False)}")
    df_agg.to_csv(os.path.join(OUTDIR, 'summary.csv'), index=False)

    # Per-seed table
    logging.info(f"\n{'='*70}\nPER-SEED RESULTS\n{'='*70}")
    for mn in model_names:
        row = f"{mn:14s}"
        for s in SEEDS:
            sub = [r for r in all_results if r['model']==mn and r['seed']==s]
            if sub: row += f" | seed={s}: {sub[0]['rmse']:.2f}"
        logging.info(row)

    config = {'region': REGION, 'ws': WS, 'seeds': SEEDS, 'epochs': N_EP,
              'bs': BS, 'models': model_names,
              'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    with open(os.path.join(OUTDIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    logging.info(f"\n{'='*60}\nALL DONE: {REGION} ws={WS}\nResults saved to: {OUTDIR}\n{'='*60}")
    logging.info(f"""
Saved files:
  {OUTDIR}/metrics.csv       — RMSE/MAE/Improv per model per seed
  {OUTDIR}/summary.csv       — Aggregated statistics
  {OUTDIR}/arrays.pkl        — All prediction arrays for figures:
    Per model-seed: pred, true, pos, err_grid, pred_grid, true_grid,
                    attn, spectral_gates, train_loss, val_loss
  {OUTDIR}/gebco_grid.npy    — GEBCO reference grid
  {OUTDIR}/truth_grid.npy    — Ground truth grid
  {OUTDIR}/mask.npy          — Valid pixel mask
  {OUTDIR}/dshape.npy        — Grid dimensions
  {OUTDIR}/config.json       — Experiment configuration
""")
