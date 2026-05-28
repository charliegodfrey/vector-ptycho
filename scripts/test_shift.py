#!/usr/bin/env python3
"""Quick test for _shift_complex_image in vector_ptycho.utils
Saves amplitude/phase images and prints center-of-mass shifts.
"""
from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / 'src'
sys.path.insert(0, str(src_dir))

import torch
import numpy as np
import matplotlib.pyplot as plt

from vector_ptycho.utils import _shift_complex_image


def make_probe(H, W, sigma=8.0):
    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    R2 = (X - cx) ** 2 + (Y - cy) ** 2
    amp = torch.exp(-0.5 * R2 / (sigma**2))
    phase = 0.5 * (X - cx) / (cx + 1.0)  # gentle phase ramp
    probe = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase))
    return probe


def com_xy(img):
    """Center-of-mass (y, x) of intensity |img|^2 in pixel coordinates."""
    I = torch.abs(img) ** 2
    H, W = I.shape
    ys = torch.arange(H, dtype=I.dtype)
    xs = torch.arange(W, dtype=I.dtype)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    tot = I.sum()
    return ( (I * Y).sum() / tot, (I * X).sum() / tot )


if __name__ == '__main__':
    H, W = 64, 64
    probe = make_probe(H, W)

    # shifts in pixels (dy, dx). _shift_complex_image expects pixel shifts.
    shifts = torch.tensor([
        [0.0, 0.0],
        [0.5, -0.3],
        [2.0, 0.0],
        [-3.0, 4.0],
    ], dtype=torch.float32)

    shifted = _shift_complex_image(probe, shifts)

    out_dir = repo_root / 'artifacts' / 'shift_test'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save original amplitude/phase
    amp0 = torch.abs(probe).cpu().numpy()
    ph0 = np.angle(probe.cpu().numpy())
    plt.imsave(out_dir / 'probe_amp_orig.png', amp0, cmap='magma')
    plt.imsave(out_dir / 'probe_phase_orig.png', ph0, cmap='hsv')

    com0 = com_xy(probe)
    print(f'Original COM (y,x) = ({com0[0].item():.3f}, {com0[1].item():.3f})')

    for i, s in enumerate(shifts):
        img = shifted[i]
        amp = torch.abs(img).cpu().numpy()
        ph = np.angle(img.cpu().numpy())
        plt.imsave(out_dir / f'probe_amp_shift_{i}.png', amp, cmap='magma')
        plt.imsave(out_dir / f'probe_phase_shift_{i}.png', ph, cmap='hsv')

        com_i = com_xy(img)
        dy_pix = com_i[0].item() - com0[0].item()
        dx_pix = com_i[1].item() - com0[1].item()
        print(f'Shift index {i}: requested shift (dy,dx) = ({s[0].item():.3f}, {s[1].item():.3f})',
              f' measured COM shift (dy,dx) = ({dy_pix:.3f}, {dx_pix:.3f})')

    # Sanity check: integer-shift comparison with torch.roll for the integer-shift entry
    int_shift = shifts[2].to(torch.int64)
    rolled = torch.roll(probe, shifts=(int_shift[0].item(), int_shift[1].item()), dims=(0,1))
    diff = torch.abs(shifted[2] - rolled).abs().max()
    print(f'Max absolute difference between grid_sample shift and torch.roll for integer shift: {diff.item():.6e}')

    print('Saved images to', out_dir)
