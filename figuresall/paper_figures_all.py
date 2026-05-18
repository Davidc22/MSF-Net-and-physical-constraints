"""
paper_figures_all.py — All paper figures (except Fig1 study area & Fig2 architecture)

Generates:
  Fig3_RMSE_heatmap.pdf       — 8 models × 4 regions improvement heatmap
  Fig4_window_size.pdf         — Window size ablation dual panel
  Fig5_psd.pdf                 — PSD error analysis (Xisha)
  Fig6_physics_heatmap.pdf     — Implicit physics ΔRMSE heatmap
  Fig7_stability.pdf           — Training stability (cross-seed std)
  Fig8_loss_curves.pdf         — Validation loss curves

All figures: Arial font, 300 DPI, PDF + PNG output.

Usage:
  python paper_figures_all.py
  python paper_figures_all.py --basedir /path/to/results --outdir figures
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
import pickle, os, argparse

# ============================================================
# GLOBAL STYLE — Arial, academic
# ============================================================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ============================================================
# DATA
# ============================================================
REGIONS = ["Xisha", "Caribbean", "Mexico", "Pacific"]
REGION_DIRS = ["xisha", "carribean", "mexico", "pac"]
SEEDS = [42, 2024, 577]
BEST_SEED = 42

# Table 3 RMSE (mean of 3 seeds)
TABLE3_RMSE = {
    "MLP":         [85.02, 191.51, 176.47, 254.31],
    "CNN":         [61.76, 151.49,  74.79,  78.58],   
    "Transformer": [34.83,  88.19,  67.75,  71.35],
    "SA":          [36.52,  86.53,  67.91,  65.72],   
    "ES-MHSA":    [34.85,  81.98,  65.32,  63.68],    
    "TransUNet":   [37.23,  91.27,  65.40,  61.61],
    "FA":          [33.98,  83.80,  64.06,  61.90],
    "MSF-Net":     [32.30,  80.92,  64.83,  62.38],
}
TABLE3_STD = {
    "MLP":         [15.01, 18.41, 18.16, 41.40],
    "CNN":         [ 2.87,  3.59,  0.72,  0.41],
    "Transformer": [ 0.98,  1.61,  0.76,  2.28],
    "SA":          [ 0.42,  1.19,  0.53,  0.11],
    "ES-MHSA":    [ 0.39,  1.23,  0.25,  0.20],
    "TransUNet":   [ 2.58,  3.45,  0.86,  0.59],
    "FA":          [ 0.17,  1.54,  0.11,  0.45],
    "MSF-Net":     [ 0.48,  0.87,  0.45,  0.39],
}
GEBCO_RMSE = [134.21, 102.11, 70.96, 70.13]
SIO_V251 = [144.11, 127.09, 89.58, 107.74]
SIO_V271 = [139.86, 126.75, 89.59, 107.55]

MODELS_8 = ["MLP", "CNN", "Transformer", "SA", "ES-MHSA", "TransUNet", "FA", "MSF-Net"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--basedir', type=str, default='.', help='Dir with results_paper_* folders')
    p.add_argument('--outdir', type=str, default='figures')
    return p.parse_args()

def load_arrays(basedir):
    """Load arrays.pkl from all 4 regions."""
    data = {}
    for rdir, rname in zip(REGION_DIRS, REGIONS):
        path = os.path.join(basedir, f'results_paper_{rdir}', 'arrays.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data[rname] = pickle.load(f)
            print(f"  Loaded {rname}: {len(data[rname])} entries")
        else:
            print(f"  WARNING: {path} not found")
    return data

def savefig(fig, name, outdir):
    fig.savefig(os.path.join(outdir, f'{name}.pdf'), format='pdf')
    fig.savefig(os.path.join(outdir, f'{name}.png'), format='png')
    plt.close(fig)
    print(f"  Saved {name}.pdf/.png")


# ============================================================
# Fig 3: Architecture RMSE Heatmap
# ============================================================
def plot_fig3(outdir):
    models = MODELS_8
    regions = REGIONS

    # Compute improvement over GEBCO (%)
    improv = np.zeros((len(models), len(regions)))
    for i, m in enumerate(models):
        for j in range(len(regions)):
            improv[i, j] = (GEBCO_RMSE[j] - TABLE3_RMSE[m][j]) / GEBCO_RMSE[j] * 100

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Use a sequential blue colormap — dark bg with white text is always readable
    cmap = plt.cm.YlGnBu
    vmin, vmax = -50, 80
    norm = Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(improv, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)

    # Cell annotations
    for i in range(len(models)):
        for j in range(len(regions)):
            r = TABLE3_RMSE[models[i]][j]
            imp = improv[i, j]
            # Dark text on light bg, white text on dark bg
            color = 'white' if imp > 40 else 'black'
            ax.text(j, i - 0.13, f'{r:.1f} m',
                    ha='center', va='center', fontsize=9.5,
                    fontweight='bold', color=color)
            ax.text(j, i + 0.20, f'{imp:.1f}%',
                    ha='center', va='center', fontsize=8.5,
                    color=color, fontstyle='italic')

    # GEBCO / SIO reference at bottom
    for j in range(len(regions)):
        ax.text(j, len(models) - 0.15,
                f'GEBCO: {GEBCO_RMSE[j]:.0f} m\nSIO v27.1: {SIO_V271[j]:.0f} m',
                ha='center', va='top', fontsize=7.5, color='#555555',
                fontstyle='italic')

    # Thin white grid
    for i in range(len(models) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.8)
    for j in range(len(regions) + 1):
        ax.axvline(j - 0.5, color='white', linewidth=0.8)

    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label('Improvement over GEBCO (%)', fontsize=11)

    ax.set_title('Cross-Region Architecture Comparison', fontsize=14, fontweight='bold', pad=12)

    plt.tight_layout()
    savefig(fig, 'Fig3_RMSE_heatmap', outdir)


# ============================================================
# Fig 4: Window Size Ablation
# ============================================================
def plot_fig4(outdir):
    ws_data = {7: (43.85, 44.78), 11: (38.78, 39.19), 15: (36.33, 36.55), 21: (34.63, 34.46)}
    ws_list = sorted(ws_data.keys())
    fa_vals = [ws_data[w][0] for w in ws_list]
    prop_vals = [ws_data[w][1] for w in ws_list]
    deltas = [ws_data[w][1] - ws_data[w][0] for w in ws_list]
    ws_labels = [f'{w}×{w}' for w in ws_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: RMSE curves
    ax1.plot(ws_labels, fa_vals, 'o-', color='#4472C4', label='FA', linewidth=2.2, markersize=8)
    ax1.plot(ws_labels, prop_vals, 's-', color='#E74C3C', label='MSF-Net', linewidth=2.2, markersize=8)
    ax1.set_ylabel('RMSE (m)')
    ax1.set_xlabel('Window Size')
    ax1.set_title('(a) RMSE vs Window Size', fontweight='bold')
    ax1.legend(framealpha=0.9)
    ax1.grid(alpha=0.3)

    # Right: Delta bars
    colors_d = ['#E74C3C' if d > 0 else '#27AE60' for d in deltas]
    bars = ax2.bar(ws_labels, deltas, color=colors_d, edgecolor='black', linewidth=0.5, width=0.5)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel('$\\Delta$ RMSE (MSF-Net $-$ FA, m)')
    ax2.set_xlabel('Window Size')
    ax2.set_title('(b) Frequency Module Benefit', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Value labels — small font, tight to bars
    for i, (bar, di) in enumerate(zip(bars, deltas)):
        y_pos = di + (0.015 if di > 0 else -0.015)
        va = 'bottom' if di > 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 f'{di:+.2f}', ha='center', va=va, fontsize=7.5, fontweight='bold')

    # Ensure y-axis has margin for labels
    ymin = min(deltas) - 0.15
    ymax = max(deltas) + 0.15
    ax2.set_ylim(ymin, ymax)

    plt.tight_layout()
    savefig(fig, 'Fig4_window_size', outdir)


# ============================================================
# Fig 5: PSD Error Analysis (Xisha)
# ============================================================
def compute_radial_psd(grid, dx=7408.0):
    g = grid.copy()
    g[np.isnan(g)] = 0
    ny, nx = g.shape
    F = np.fft.fft2(g)
    P = np.abs(F)**2 / (ny * nx)
    fy = np.fft.fftfreq(ny, d=dx)
    fx = np.fft.fftfreq(nx, d=dx)
    fxx, fyy = np.meshgrid(fx, fy)
    r = np.sqrt(fxx**2 + fyy**2)
    r_flat = r.flatten()
    p_flat = P.flatten()
    n_bins = min(ny, nx) // 2
    bin_edges = np.linspace(0, r_flat.max(), n_bins + 1)
    psd = np.zeros(n_bins)
    wavelengths = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r_flat >= bin_edges[i]) & (r_flat < bin_edges[i+1])
        if mask.sum() > 0:
            psd[i] = p_flat[mask].mean()
            wavelengths[i] = 1.0 / ((bin_edges[i] + bin_edges[i+1]) / 2 + 1e-20) / 1000
    valid = (wavelengths > 0) & (wavelengths < 500) & (psd > 0)
    return wavelengths[valid], psd[valid]


def plot_fig5(arrays_data, outdir):
    region = 'Xisha'
    if region not in arrays_data:
        print("  WARNING: Xisha arrays not found, skipping Fig5")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    models_psd = ["MLP", "CNN", "ES-MHSA", "TransUNet", "FA", "Proposed"]
    colors_psd = ['#BBBBBB', '#888888', '#5DADE2', '#E67E22', '#F4D03F', '#E74C3C']
    linewidths = [1.2, 1.2, 1.8, 1.8, 2.2, 2.8]
    labels_psd = ["MLP", "CNN", "ES-MHSA", "TransUNet", "FA", "MSF-Net"]

    for mi, m in enumerate(models_psd):
        key = f"{m}_seed{BEST_SEED}"
        if key not in arrays_data[region]:
            continue
        err_grid = arrays_data[region][key].get('err_grid')
        if err_grid is None:
            continue
        wl, psd = compute_radial_psd(err_grid)
        if len(wl) == 0:
            continue
        ax.loglog(wl, psd, color=colors_psd[mi], linewidth=linewidths[mi],
                  label=labels_psd[mi], alpha=0.9)

    # True signal PSD — try loading truth_grid
    truth_path = None
    for rdir in REGION_DIRS:
        tp = os.path.join('.', f'results_paper_{rdir}', 'truth_grid.npy')
        if 'xisha' in rdir and os.path.exists(tp):
            truth_path = tp
            break
    if truth_path:
        true_grid = np.load(truth_path)
        wl_t, psd_t = compute_radial_psd(true_grid)
        if len(wl_t) > 0:
            ax.loglog(wl_t, psd_t, '--', color='grey', linewidth=1.5,
                      label='True signal', alpha=0.6)

    # 15-160 km band
    ax.axvspan(15, 160, alpha=0.07, color='green')
    ylim = ax.get_ylim()
    ax.text(48, ylim[1] * 0.3, '15–160 km\ncorrelation band',
            fontsize=9, ha='center', va='top', color='#27AE60', fontstyle='italic')

    ax.set_xlabel('Wavelength (km)')
    ax.set_ylabel('Error PSD')
    ax.set_title('Radially Averaged Error PSD (Xisha)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25, which='both')
    ax.set_xlim(5, 400)
    ax.invert_xaxis()

    plt.tight_layout()
    savefig(fig, 'Fig5_psd', outdir)


# ============================================================
# Fig 6: Implicit Physics Heatmap
# ============================================================
def plot_fig6(outdir):
    implicit_base = {"Xisha": 32.70, "Caribbean": 81.95, "Mexico": 64.45, "Pacific": 62.70}
    implicit = {
        "+MTL":     {"Xisha": 33.50, "Caribbean": 79.45, "Mexico": 63.88, "Pacific": 61.94},
        "+SPI":     {"Xisha": 32.95, "Caribbean": 80.67, "Mexico": 64.43, "Pacific": 62.71},
        "+PFA":     {"Xisha": 33.58, "Caribbean": 81.56, "Mexico": 64.09, "Pacific": 62.80},
        "+MTL+SPI": {"Xisha": 33.99, "Caribbean": 79.92, "Mexico": 63.97, "Pacific": 62.26},
    }
    configs = list(implicit.keys())
    regions = REGIONS

    delta = np.array([[implicit[c][r] - implicit_base[r] for r in regions] for c in configs])

    fig, ax = plt.subplots(figsize=(8, 3.8))
    vmax = max(abs(delta.min()), abs(delta.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(delta, cmap='RdBu_r', norm=norm, aspect='auto')

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=11)

    for i in range(len(configs)):
        for j in range(len(regions)):
            val = delta[i, j]
            color = 'white' if abs(val) > vmax * 0.55 else 'black'
            ax.text(j, i, f'{val:+.2f}', ha='center', va='center', fontsize=10,
                    fontweight='bold' if abs(val) > 0.5 else 'normal', color=color)

    # GEBCO Improv% at bottom
    #gebco_improv = [75.6, 19.7, 9.2, 10.6]
    #for j, v in enumerate(gebco_improv):
    #    ax.text(j, len(configs) + 0.12, f'Improv={v:.0f}%', ha='center', va='top',
    #            fontsize=8.5, color='#666666', fontstyle='italic')

    #cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    #cbar.set_label('$\\Delta$RMSE (m)', fontsize=11)

    ax.set_title('Implicit Physics: RMSE Change vs Baseline', fontsize=13, fontweight='bold', pad=10)

    plt.tight_layout()
    savefig(fig, 'Fig6_physics_heatmap', outdir)


# ============================================================
# Fig 7: Training Stability
# ============================================================
def plot_fig7(outdir):
    models_cmp = ["MSF-Net", "TransUNet", "FA", "ES-MHSA"]
    model_keys = ["MSF-Net", "TransUNet", "FA", "ES-MHSA"]
    colors_cmp = ['#E74C3C', '#3498DB', '#F39C12', '#85C1E9']

    stds = {}
    for m, mk in zip(models_cmp, model_keys):
        stds[m] = TABLE3_STD.get(mk, [0]*4)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(REGIONS))
    width = 0.19

    for i, m in enumerate(models_cmp):
        offset = (i - 1.5) * width
        ax.bar(x + offset, stds[m], width, label=m, color=colors_cmp[i],
               edgecolor='black', linewidth=0.5, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS, fontsize=11, fontweight='bold')
    ax.set_ylabel('Cross-Seed Std (m)')
    ax.set_title('Training Stability: Cross-Seed Standard Deviation', fontweight='bold')
    ax.legend(ncol=4, fontsize=9, loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    savefig(fig, 'Fig7_stability', outdir)


# ============================================================
# Fig 8: Training Loss Curves
# ============================================================
def plot_fig8(arrays_data, outdir):
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    models_loss = ["Proposed", "TransUNet", "FA"]
    labels_loss = ["MSF-Net", "TransUNet", "FA"]
    colors_loss = ['#E74C3C', '#3498DB', '#F39C12']

    for idx, (region, rdir) in enumerate(zip(REGIONS, REGION_DIRS)):
        ax = axes[idx]
        if region not in arrays_data:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(region, fontweight='bold')
            continue

        for mi, (m, label) in enumerate(zip(models_loss, labels_loss)):
            key = f"{m}_seed{BEST_SEED}"
            if key not in arrays_data[region]:
                continue
            val_loss = arrays_data[region][key].get('val_loss', [])
            if len(val_loss) == 0:
                continue
            ax.plot(val_loss, color=colors_loss[mi], linewidth=1.3, label=label, alpha=0.85)

        ax.set_title(region, fontweight='bold')
        ax.set_xlabel('Epoch')
        if idx == 0:
            ax.set_ylabel('Val Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize=8, framealpha=0.9)

    plt.suptitle('Validation Loss Curves', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    savefig(fig, 'Fig8_loss_curves', outdir)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("="*60)
    print("Generating all paper figures (except Fig1 & Fig2)")
    print("="*60)

    # Load arrays for PSD and loss curve figures
    print("\nLoading experiment arrays...")
    arrays_data = load_arrays(args.basedir)

    print(f"\nSaving to {args.outdir}/")
    print("-"*60)

    plot_fig3(args.outdir)       # RMSE heatmap
    plot_fig4(args.outdir)       # Window size ablation
    plot_fig5(arrays_data, args.outdir)  # PSD
    plot_fig6(args.outdir)       # Physics heatmap
    plot_fig7(args.outdir)       # Stability
    plot_fig8(arrays_data, args.outdir)  # Loss curves

    print(f"\n{'='*60}")
    print("All figures generated.")
    print(f"{'='*60}")
    print("""
Paper figure mapping:
  Fig 1:  Study area map         — use paper_fig1_fig3.py separately
  Fig 2:  Architecture diagram   — use Napkin AI / draw.io
  Fig 3:  Fig3_RMSE_heatmap      — 8-model × 4-region improvement heatmap
  Fig 4:  Fig4_window_size       — Window size ablation dual panel
  Fig 5:  Fig5_psd               — PSD error analysis (Xisha)
  Fig 6:  Fig6_physics_heatmap   — Implicit physics ΔRMSE heatmap
  Fig 7:  Fig7_stability         — Training stability comparison
  Fig 8:  Fig8_loss_curves       — Validation loss curves (Discussion)
""")
