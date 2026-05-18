# MSF-Net: Multi-Scale Spectral Fusion Network for Satellite Gravity-Derived Bathymetry

This repository contains the source code for the paper:

Multi-Scale Spectral Fusion Network for Satellite Gravity-Derived Bathymetry: Architecture Design, Physics Integration, and Cross-Region Analysis
Xingyu Chai, Hongze Leng*, Shengjun Zhang, Junqiang Song*

Overview
MSF-Net is a frequency-domain deep learning framework for reconstructing seafloor topography from satellite gravity data. It introduces three modules:

Learnable Spectral Filter (LSF): adaptive per-frequency gating on the FFT spectrum
Multi-Scale Spectral Fusion (MSF): dual-scale FFT with zero-padding for enhanced spectral resolution
Cross-Channel Spectral Interaction (CCSI): lightweight cross-channel calibration (10 parameters)

The framework is validated across four ocean regions (Xisha, Caribbean, Gulf of Mexico, Pacific) against eight baseline architectures and two satellite gravity products (SIO v25.1/v27.1).
Repository Structure
├── 8modelcomparison/            # Cross-region architecture comparison
│   ├── MSFNet_modelcomparison_xisha.py
│   ├── MSFNet_modelcomparison_caribbean.py
│   ├── MSFNet_modelcomparison_mexico.py
│   ├── MSFNet_modelcomparison_pac.py
│   └── run_all.sh               # Shell script for parallel execution
│
├── ablation_and_physicalconstraints/  # Ablation and physics integration
│   └── FA_integrated_final.py   # Component ablation + explicit/implicit physics
│
├── data/                        # Input data (satellite gravity + ground truth)
│
└── figuresall/                  # Figure generation
    └── paper_figures_all.py     # All paper figures (Fig 3–8)
Experiments
Cross-Region Model Comparison (Table 3)
Compares 8 architectures (MLP, CNN, Transformer, SA, ES-MHSA, TransUNet, FA, MSF-Net) across 4 regions with 3 random seeds:
bashcd 8modelcomparison
bash run_all.sh
Architecture Ablation and Physics Constraints (Tables 1, 2, 4, 5)
Runs component ablation, window-size ablation, and explicit/implicit physics experiments:
bashcd ablation_and_physicalconstraints
python FA_integrated_final.py --region xisha --ws 21 --seeds 42 123 7
Figure Generation
Generates all paper figures from experiment outputs:
bashcd figuresall
python paper_figures_all.py --basedir ../8modelcomparison --outdir figures
Requirements

Python ≥ 3.8
PyTorch ≥ 1.12
NumPy, SciPy, Pandas, Matplotlib, scikit-learn, tqdm

Data
Input data consists of five channels at 1 arc-minute resolution:
ChannelSourceGravity Anomaly (GA)Sandwell group V32 + SWOTVertical Gravity Gradient (VGG)Sandwell group V32 + SWOTDeflection of the Vertical, North (DOV-N)Sandwell group V32 + SWOTDeflection of the Vertical, East (DOV-E)Sandwell group V32 + SWOTPrior Bathymetry (GEBCO)GEBCO_2024
Ground truth: multi-beam echosounder measurements from GMGS (Xisha) and NCEI (other regions).
Citation
If you find this work useful, please cite:
bibtex@article{chai2025msfnet,
  title={Multi-Scale Spectral Fusion Network for Satellite Gravity-Derived Bathymetry: Architecture Design, Physics Integration, and Cross-Region Analysis},
  author={Chai, Xingyu and Leng, Hongze and Zhang, Shengjun and Song, Junqiang},
  journal={Under Review},
  year={2025}
}
License
This project is released under the MIT License.
