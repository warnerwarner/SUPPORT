# SUPPORT — Self-Supervised Voltage Imaging Denoising

## Basics

- **Full name:** SUPPORT (Statistically Unbiased Prediction utilizing SpatiOtemporal information in imaging daTa)
- **Purpose:** Self-supervised deep learning denoising for voltage imaging data. No ground truth needed — trains on noisy data alone.
- **Published:** Nature Methods, 2023 (Eom et al., NICALab/KAIST)
- **Paper:** https://www.nature.com/articles/s41592-023-02005-8
- **Upstream repo:** https://github.com/NICALab/SUPPORT
- **My fork:** Local at `~/data/support/` (same machine as Claude). Previously on `gpu-2001` accessed remotely via SSH. This CLAUDE.md was written by agents on remote machines connecting via SSH.

## Access

- **Now local** — Claude runs on same machine as this project. No SSH needed.
- Path: `~/data/support/` (resolves to `/gpfs/home/warnet02/data/support/`)
- **Shared data:** `/gpfs/data/shohamlab/tom/`
- No destructive commands without confirmation

## Architecture

Two-stage network: U-Net + Blind-Spot Network (BSN).

### U-Net (spatiotemporal context)
- Input: all non-center temporal frames → encode/decode with skip connections
- Output: 16-channel feature map used to guide BSN via residual addition in first layer

### Blind-Spot Network (center frame denoising)
- Two parallel dilated-convolution paths:
  - **3x3 path:** 6 layers, dilations [1, 2, 4, 8, 16, 32], depth=6
  - **5x5 path:** 4 layers, dilations [1, 3, 9, 27], depth=4
- **ConvHole2D** (`model/convhole.py`): custom conv with center weight fixed at zero — cannot use center pixel = blind-spot enforced
- **Skip injection:** center frame re-injected at each BSN layer via learned scalar (no phase) or 1x1 conv projection (with phase)
- Final concatenation: last 3x3 output + last 5x5 output

### Combined output
- U-Net output + BSN output concatenated → 1x1 convs → single-channel prediction

## My Custom Extensions

### Phase Conditioning (`--use_phase_conditioning`)
- Extracts phase from sinusoidal position signal via Hilbert transform
- Phase sin/cos channels added to U-Net input (per-frame, broadcast spatially) and BSN input (center frame, per-row)
- Controlled via `model/film.py` (FiLM layers) and direct concatenation
- Docs: `PHASE_CONDITIONING_*.md`, `INDEX_PHASE_CONDITIONING.md`

### Splatting Stage (`--use_splatting`)
- Takes raw EOD + position data `[B, T, 2, H, W]` instead of pre-aligned frames
- GPU splatting: interpolates from scan coordinates to regular grid
- `model/splatting_stage.py`: `LearnableSplattingStage`
- Docs: `docs/ANISOTROPIC_IMPLEMENTATION.md`

### Point Offset Stage (`--use_point_offset`)
- Learnable per-pixel offsets applied before splatting
- Corrects residual motion/scan distortions
- `model/point_offset_stage.py`: `PointOffsetStage`
- When active, splatting parameters frozen initially

### Multi-field SUPPORT (`model/multiFieldSUPPORT.py`)
- Variant for multi-field-of-view data

### Distributed Training
- Multi-GPU single-node: `train_distributed_single_node.sh`
- Multi-GPU multi-node: `train_distributed_multi_node.sh`, `train_distributed_multi_node_gpu4.sh`, `train_distributed_multi_node_gl40s.sh`
- Sweeps: `launch_sweep.sh`, `train_sweep_job.sh`
- Monitoring: `monitor_training.sh`

### Zarr & Lazy Loading
- `tiff_to_zarr.py`: converts TIFF stacks to zarr for efficient access
- `src/utils/alignedzarr.py`: AlignedZarr class for reading aligned zarr data
- Lazy loading for large datasets that don't fit in RAM

### GUI
- PyQt5-based: `src/GUI/test_GUI.py`, `src/GUI/train_GUI.py`

## Tech Stack

- **Package manager:** pixi (pixi.toml, pixi.lock)
- **Environment:** conda-forge, Python 3.10, PyTorch GPU, CUDA 13
- **Key deps:** numpy, scikit-image, zarr, tqdm, tensorboard, matplotlib, opencv-python, numba, pyfftw
- **Dev:** pytest, ruff, basedpyright
- **Cluster:** slurm (sbatch scripts, HPC environment)

## Data Format

- **TIFF:** original format, read via tifffile
- **Zarr:** converted format for lazy loading (`--is_zarr`)
- **Raw mode:** unaligned data with EOD and position channels (`--is_raw`)
- **Shape:** typically `[T, H, W]` uint16, frame rate ~440Hz
- **Sample data:** `data/sample_data.tif`

## Key Training Parameters

| Flag | Description | Typical |
|------|-------------|---------|
| `--in_channels` | Temporal context window size | 61 |
| `--bs_size` | Blind spot size [H, W] | [1, 3] or [3, 3] |
| `--n_epochs` | Training epochs | 10-50 |
| `--loss_option` | l1_l2 / huber / robust | l1_l2 |
| `--loss_coef` | [L1_weight, L2_weight] | [0.5, 0.5] |
| `--is_raw` | Use raw unaligned data | varies |
| `--is_zarr` | Use zarr format | varies |
| `--is_folder` | Train on directory of TIFFs | varies |
| `--use_phase_conditioning` | Enable phase conditioning | varies |
| `--use_splatting` | Enable splatting stage | varies |
| `--use_point_offset` | Enable point offset learning | varies |

## Training History

Main dataset: **stephenVoltage** (Stephen's voltage imaging recordings).

Many experiments exploring:
- Blind spot sizes: 1x3, 2x2, 3x3 with/without phase conditioning
- Context window sizes (l105l205 → lookback 105, lookahead 205)
- Injection vs no-injection
- Wider models (1.5x channel multiplier)
- Multi-node distributed training
- Gold standard: isotropic dilations (no anisotropic)

Saved models in `results/saved_models/<experiment>/model_X.pth`.

## File Map

| Path | Purpose |
|------|---------|
| `model/SUPPORT.py` | Main model definition |
| `model/convhole.py` | ConvHole2D/3D — blind-spot convolution |
| `model/splatting_stage.py` | Learnable GPU splatting |
| `model/point_offset_stage.py` | Learnable point offsets |
| `model/film.py` | FiLM conditioning layers |
| `model/multiFieldSUPPORT.py` | Multi-FOV variant |
| `src/train.py` | Main training loop |
| `src/train_distributed.py` | Distributed training entry |
| `src/test.py` | Inference |
| `src/test_splatting.py` | Splatting inference |
| `src/utils/dataset.py` | Dataset/dataloader + phase extraction |
| `src/utils/util.py` | Argument parsing, helpers |
| `src/utils/alignment.py` | Data alignment (motion correction) |
| `src/utils/alignedzarr.py` | Aligned zarr reader |
| `src/utils/splatting_loss.py` | Loss for splatting parameters |
| `src/utils/splatting_validation.py` | Splatting validation + viz |
| `tiff_to_zarr.py` | TIFF → zarr converter |
| `src/GUI/train_GUI.py` | PyQt5 training GUI |
| `src/GUI/test_GUI.py` | PyQt5 inference GUI |
| `docs/README.md` | Original upstream docs |
| `docs/Beginner_guide.md` | Beginner setup guide |
| `pyproject.toml` | Project config + pixi deps |

## Current Problem: Raw vs Reconstructed Data

### Data pipeline (from actual inspection)

- **Raw acquired data:** `~/data/stephen/zarr` on gpu-2001
  - Shape: ~2200 x 32 x 3606 (T, Y, X) — 32 rows, ~3606 samples per row
  - Each zarr: `eod` (acquired signal, int16) and `position` (oscillator position, int16)
  - Position sawtooth waveform (~0 to ~20891), one per row
  - Some have `reconstructed` (earlier recon) or `splatted` (newer recon)
  - `reconstructed` shape: ~(T, 160, 480) or ~(T, 160, 350)
  - `splatted` shape: ~(T, 480, 1440) — only in 2 files (run011: 00060, 00072)
- **Acquisition:** not normal 2P raster scan — oscillating EOD scanner
  - Each row scanned at different oscillator position
  - Phase offset changes for every row in every frame
  - Same pixel coordinate in frame 1 vs frame 2 → different spatial locations

### Three approaches, all with issues

**Approach A: Train on reconstructed data**
- Complex reconstruction maps 32 raw rows → ~160 spatial rows
- Reconstruction causes bleeding between pixels
- SUPPORT exploits cross-pixel correlations from recon → breaks blind-spot independence
- Poor denoising result

**Approach B: Train on raw data + alignment (`align_data` / `AlignedZarr`)**
- `align_data` cross-correlates position signals to find per-row per-frame shifts
- Crops to common overlap region (min_length after shifts)
- Simple alignment, doesn't account for oscillator position → spatial location mapping
- Without phase info: SUPPORT applies strong spatial smoothing (averaging over oscillator)
- Loses high spatial fidelity

**Approach C: Train on raw data + phase conditioning (`--use_phase_conditioning`)**
- Extracts per-row phase from position via Hilbert transform → sin/cos
- Phase sin/cos concatenated to U-Net input (per-frame, broadcast spatially) and BSN input (center frame, per-row)
- Model sees phase info but does NOT use it to warp/register data spatially
- Limited success — model still treats rows as fixed spatial locations

**Approach D: Train on raw data + splatting (`--use_splatting`)**
- Most principled approach — physically maps each sample to correct spatial location
- `LearnableSplattingStage` takes [B, T, 2, H, W] where dim 2 = [eod, position]
- `_osc_matched_filt_torch` extracts oscillation frequency from position
- `_build_scan_model` creates scanner trajectory model (v, u coordinates)
- Adds oscillation modulation to v-coordinates (EOD deflection)
- Per-point splatting to output grid + Gaussian blur
- 8 learnable parameters (amplitude, phase shifts, sigma, etc.)
- Only applied to 2 files so far

### Goal

Denoise raw data while preserving spatial fidelity. Account for oscillator phase offset so SUPPORT knows which spatial locations correspond across frames. Splatting is the most promising approach.

## Common Tasks

- **Train:** `python -m src.train --exp_name <name> --noisy_data <path> --n_epochs <N>`
- **Inference:** edit `src/test.py` → `python -m src.test`
- **Distributed:** `bash train_distributed_multi_node.sh`
- **Zarr convert:** `python tiff_to_zarr.py`
- **Monitor training:** `bash monitor_training.sh`
