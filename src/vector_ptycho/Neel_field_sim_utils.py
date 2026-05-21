import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import os

from vector_ptycho.utils import *
from vector_ptycho.plotting_utils import *

def generate_circle(radius=0.5e-6, Nx=300,Ny=300, Lx=5e-6, Ly=5e-6, plot=True, cm='twilight', device=None):
    # --- Grid ---

    x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
    y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='xy')

    R = np.sqrt(X**2 + Y**2)
    #Mz = np.tanh((R - radius) / (0.25e-6)).astype(np.float32)
    Mz = 1.0*(R > radius)
    Mz = np.clip(Mz, -0.9999, 0.9999)

    Minplane = np.sqrt(np.maximum(1.0 - Mz**2, 0.0)).astype(np.float32)
    phi_inplane = np.arctan2(Y, X)
    #phi_inplane = 0.0
    Mx = Minplane * np.cos(phi_inplane) * np.exp(-R**2 / (2 * (5e-6)**2))
    My = Minplane * np.sin(phi_inplane) * np.exp(-R**2 / (2 * (5e-6)**2))

    norm3 = np.sqrt(Mx**2 + My**2 + Mz**2)
    norm3[norm3 == 0] = 1.0
    Mx = Mx / norm3
    My = My / norm3
    Mz = Mz / norm3

    theta_np = np.arccos(np.clip(Mz, -1.0, 1.0)).astype(np.float32)
    phi_np   = np.mod(np.arctan2(My, Mx), 2.0 * np.pi).astype(np.float32)

    if plot:
        fig, axes = plot_theta_phi_maps(theta_np, phi_np, Lx, Ly,
                           positions=None,
                           theta_cmap='magma',
                           phi_cmap=cm,
                           dx=0.0, dy=0.0,
                           show_positions=False,
                           label_positions=False,
                           label_axes=False)
        plt.tight_layout()
        plt.show()

    theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=device)
    phi_torch   = torch.tensor(phi_np, dtype=torch.float32, device=device)

    return theta_torch, phi_torch, Mx, My, Mz

def make_meron_antimeron_theta_phi(
    Nx=300,
    Ny=300,
    Lx=10.0,
    Ly=None,
    r1=(-0.8e-6, 0.0),
    r2=(0.8e-6, 0.0),
    sigma=0.25e-6,
    m0=0.95,
    phi0=np.pi / 2,
    align_sigma=5.0,
    quiver_step=10,
    plot=True,
    save_path='Meron-anti-meron-quiver.svg',
    export_path=None,
    return_torch=True,
    out_device=None,
    cm = 'twilight'
):
    """
    Build a meron-antimeron texture and return (theta, phi) suitable for NeelObject.build_jones().

    Returns:
        theta, phi with shape (Ny, Nx). By default returns torch tensors.
        mx, My, Mz arrays of shape (Ny, Nx) for optional further analysis or plotting.
    """
    if out_device is None:
        out_device = "cuda" if torch.cuda.is_available() else "cpu"

    if Ly is None:
        Ly = Lx

    r1 = np.asarray(r1, dtype=np.float32)
    r2 = np.asarray(r2, dtype=np.float32)

    # --- Grid ---
    x = np.linspace(-Lx/2, Lx/2, Nx, dtype=np.float32)
    y = np.linspace(-Ly/2, Ly/2, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='xy')

    def angle_vortex(Xg, Yg, r0):
        return np.arctan2(Yg - r0[1], Xg - r0[0])

    def gauss_core(Xg, Yg, r0, s):
        return np.exp(-((Xg - r0[0])**2 + (Yg - r0[1])**2) / (2.0 * s**2))

    def gauss_alignment_to_background(Xg, Yg, s):
        return np.exp(-(Xg**2 + Yg**2) / (2.0 * s**2))

    # --- Meron-antimeron texture ---
    phi_inplane = angle_vortex(X, Y, r1) - angle_vortex(X, Y, r2) + phi0
    Mz = m0 * (gauss_core(X, Y, r1, sigma) - gauss_core(X, Y, r2, sigma))
    Mz = np.clip(Mz, -0.9999, 0.9999)

    Minplane = np.sqrt(np.maximum(1.0 - Mz**2, 0.0))
    Mx = Minplane * np.cos(phi_inplane) * gauss_alignment_to_background(X, Y, s=align_sigma)
    My = Minplane * np.sin(phi_inplane)

    # Optional renormalization to keep spin vectors numerically bounded.
    norm3 = np.sqrt(Mx**2 + My**2 + Mz**2)
    norm3[norm3 == 0] = 1.0
    Mx = Mx / norm3
    My = My / norm3
    Mz = Mz / norm3

    # Convert to spherical angles used by NeelObject.build_jones(theta, phi)
    # theta: polar angle from +z, phi: azimuth in x-y plane.
    theta_np = np.arccos(np.clip(Mz, -1.0, 1.0)).astype(np.float32)
    phi_np = np.mod(np.arctan2(My, Mx), 2.0 * np.pi).astype(np.float32)

    if export_path is not None:
        np.savez(export_path, theta=theta_np, phi=phi_np)

    if plot:
        fig, axes = plot_theta_phi_maps(theta_np, phi_np, Lx, Ly,
                           positions=None,
                           theta_cmap='magma',
                           phi_cmap=cm,
                           dx=0.0, dy=0.0,
                           show_positions=False,
                           label_positions=False,
                           label_axes=False)
        '''
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: original style plot (Mz + in-plane quiver)
        im0 = axes[0].imshow(
            Mz,
            extent=[-Lx, Lx, -Ly, Ly],
            origin='lower',
            alpha=0.8,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
        )
        cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar0.set_label(r'$\mathscr{l}_z$')

        xs = X[::quiver_step, ::quiver_step]
        ys = Y[::quiver_step, ::quiver_step]
        mxs = Mx[::quiver_step, ::quiver_step]
        mys = My[::quiver_step, ::quiver_step]

        qnorm = np.sqrt(mxs**2 + mys**2)
        qnorm[qnorm == 0] = 1.0
        axes[0].quiver(xs, ys, mxs / qnorm, mys / qnorm, scale=50)
        axes[0].set_title('Meron-antimeron texture')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        # Panel 2: theta heatmap
        im1 = axes[1].imshow(np.rad2deg(theta_np), extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap='magma', vmin=0, vmax=180)
        cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.set_label(r'$\theta$ [deg]')
        axes[1].set_title('Theta heatmap')
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        # Panel 3: phi heatmap
        im2 = axes[2].imshow(np.rad2deg(phi_np), extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap=cm, vmin=0, vmax=360)
        cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        cbar2.set_label(r'$\phi$ [deg]')
        axes[2].set_title('Phi heatmap')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        '''
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.show()

    if return_torch:
        theta = torch.tensor(theta_np, dtype=torch.float32, device=out_device)
        phi = torch.tensor(phi_np, dtype=torch.float32, device=out_device)
        return theta, phi, Mx, My, Mz

    return theta_np, phi_np, Mx, My, Mz


import numpy as np
import torch
import matplotlib.pyplot as plt

def make_ip_strip_domain_walls(
    Nx=300,
    Ny=300,
    Lx=10.0,
    Ly=None,
    x0=0.0,
    strip_width=2.0e-6,        # full width of the central IP region
    wall_width=0.25e-6,        # sharpness of each OOP–IP transition
    m0=0.95,
    wall_type='neel',          # 'neel' or 'bloch'
    chirality=+1.0,            # +1 or -1 flips in-plane direction
    wall_axis='y',             # 'y' -> wall runs along y, normal along x
    oop_config='parallel',     # 'parallel'     -> +m0 | 0 | +m0
                               # 'antiparallel' -> -m0 | 0 | +m0
    quiver_step=10,
    plot=True,
    save_path='IP-strip-domain-wall.svg',
    export_path=None,
    return_torch=True,
    out_device=None,
    cm='twilight',
):
    """
    Build a central in-plane (Mz = 0) strip sandwiched between two OOP regions.

    Mimics the domain-wall texture seen in hematite below the Morin transition
    temperature, where an easy-plane ordered strip sits between OOP domains.

    Two flanking wall transitions (at ±strip_width/2) replace the single tanh
    of the original function:

        oop_config='parallel'     :  +m0  |  IP strip  |  +m0
        oop_config='antiparallel' :  -m0  |  IP strip  |  +m0

    Returns:
        theta, phi with shape (Ny, Nx) for NeelObject.build_jones().
        Also returns Mx, My, Mz for optional analysis/plotting.
    """
    if out_device is None:
        out_device = "cuda" if torch.cuda.is_available() else "cpu"

    if Ly is None:
        Ly = Lx

    # --- Grid ---
    x = np.linspace(-Lx / 2, Lx / 2, Nx, dtype=np.float32)
    y = np.linspace(-Ly / 2, Ly / 2, Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='xy')

    # --- Wall-normal coordinate ---
    if wall_axis == 'y':
        coord = X - x0
    elif wall_axis == 'x':
        coord = Y - x0
    else:
        raise ValueError("wall_axis must be either 'x' or 'y'.")

    # --- Two-wall Mz profile ---
    # Each wall is a tanh centred at ±half_strip from x0.
    # Combining them gives an IP strip (Mz ≈ 0) between two OOP regions.
    half = strip_width / 2.0

    left_wall  = np.tanh((coord + half) / wall_width).astype(np.float32)
    right_wall = np.tanh((coord - half) / wall_width).astype(np.float32)

    if oop_config == 'parallel':
        # Both flanking domains have the same OOP sign (+m0).
        # Profile: +1 far left, dips to 0 in strip, recovers to +1 far right.
        #   left_wall:  -1 → +1   (crosses 0 at x = -half)
        #   right_wall: -1 → +1   (crosses 0 at x = +half)
        #   0.5*(left - right): +1 inside strip, 0 outside  →  invert & shift
        profile = 1.0 - 0.5 * (left_wall - right_wall)
    elif oop_config == 'antiparallel':
        # Flanking domains have opposite OOP sign (−m0 left, +m0 right).
        # Smoothly transitions -1 → 0 (strip) → +1.
        profile = 0.5 * (left_wall + right_wall)
    else:
        raise ValueError("oop_config must be 'parallel' or 'antiparallel'.")

    Mz = m0 * profile
    Mz = np.clip(Mz, -0.9999, 0.9999)

    # --- In-plane amplitude ---
    # Peaks wherever |Mz| is small, i.e. inside the IP strip and at the two
    # OOP–IP transition zones.
    Minplane = np.sqrt(np.maximum(1.0 - Mz**2, 0.0)).astype(np.float32)

    # --- In-plane direction (Néel vs Bloch) ---
    if wall_type.lower() == 'neel':
        ang = 0.0 if wall_axis == 'y' else np.pi / 2
    elif wall_type.lower() == 'bloch':
        ang = np.pi / 2 if wall_axis == 'y' else 0.0
    else:
        raise ValueError("wall_type must be either 'neel' or 'bloch'.")

    ang = ang + (0.0 if chirality >= 0 else np.pi)

    Mx = Minplane * np.cos(ang)
    My = Minplane * np.sin(ang)

    # --- Renormalise ---
    norm3 = np.sqrt(Mx**2 + My**2 + Mz**2)
    norm3[norm3 == 0] = 1.0
    Mx = Mx / norm3
    My = My / norm3
    Mz = Mz / norm3

    # --- Spherical angles ---
    theta_np = np.arccos(np.clip(Mz, -1.0, 1.0)).astype(np.float32)
    phi_np   = np.mod(np.arctan2(My, Mx), 2.0 * np.pi).astype(np.float32)

    if export_path is not None:
        np.savez(export_path, theta=theta_np, phi=phi_np)

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].imshow(
            Mz,
            extent=[-Lx / 2, Lx / 2, -Ly / 2, Ly / 2],
            origin='lower',
            cmap='coolwarm',
            vmin=-1, vmax=1,
            alpha=0.9,
        )
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=r'$m_z$')
        xs  = X[::quiver_step, ::quiver_step]
        ys  = Y[::quiver_step, ::quiver_step]
        mxs = Mx[::quiver_step, ::quiver_step]
        mys = My[::quiver_step, ::quiver_step]
        qnorm = np.sqrt(mxs**2 + mys**2)
        qnorm[qnorm == 0] = 1.0
        axes[0].quiver(xs, ys, mxs / qnorm, mys / qnorm, scale=40, color='k')
        axes[0].set_title(
            f'{wall_type.capitalize()} IP strip '
            f'({oop_config} OOP)\nstrip_width={strip_width*1e6:.2f} µm'
        )
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        im1 = axes[1].imshow(
            np.rad2deg(theta_np),
            extent=[-Lx / 2, Lx / 2, -Ly / 2, Ly / 2],
            origin='lower',
            cmap='magma',
            vmin=0, vmax=180,
        )
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label=r'$\theta$ [deg]')
        axes[1].set_title('Theta heatmap')
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        im2 = axes[2].imshow(
            np.rad2deg(phi_np),
            extent=[-Lx / 2, Lx / 2, -Ly / 2, Ly / 2],
            origin='lower',
            cmap=cm,
            vmin=0, vmax=360,
        )
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label=r'$\phi$ [deg]')
        axes[2].set_title('Phi heatmap')
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.show()

    if return_torch:
        theta = torch.tensor(theta_np, dtype=torch.float32, device=out_device)
        phi   = torch.tensor(phi_np,   dtype=torch.float32, device=out_device)
        return theta, phi, Mx, My, Mz

    return theta_np, phi_np, Mx, My, Mz