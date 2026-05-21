import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython.display import display
from datetime import datetime
import os

from vector_ptycho._shared import _to_numpy
from vector_ptycho.Neel_field_sim_utils import *

# Avoid circular import by importing reconstruction_utils only when needed

def complex_to_rgb(Z, gamma=0.7):
    """Convert complex-valued array to RGB using HSV color space.
    
    Hue = phase, Brightness = magnitude (with gamma correction)
    """
    amp = np.abs(Z)
    phase = np.angle(Z)

    amp = amp / (amp.max() + 1e-10)  # Avoid division by zero
    amp = amp**gamma

    h = (phase + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = amp

    hsv = np.stack([h, s, v], axis=-1)
    return hsv_to_rgb(hsv)


def neel_direction_to_rgb(theta, phi, gamma=0.7, phi_cmap='twilight'):
    """Convert a Néel-vector direction to RGB.

    Phi is mapped through the supplied colormap with pi-periodic wrapping,
    and brightness fades with |Lz| = |cos(theta)| so the in-plane direction
    is dimmed as the out-of-plane component grows.
    """
    lz = np.abs(np.cos(theta))
    brightness = np.clip(1.0 - lz, 0.0, 1.0)
    brightness = brightness**gamma

    phi_norm = np.mod(phi, np.pi) / np.pi
    phi_rgb = plt.get_cmap(phi_cmap)(phi_norm)[..., :3]
    return phi_rgb * brightness[..., np.newaxis]


def make_color_wheel_rgba(gamma=1.0, Nw=200):
    u = np.linspace(-1, 1, Nw)
    v = np.linspace(-1, 1, Nw)
    U, V = np.meshgrid(u, v)

    Rw = np.sqrt(U**2 + V**2)
    Phiw = np.arctan2(V, U)

    magnitude = np.clip(Rw, 0, 1)

    Zw = magnitude * np.exp(1j * Phiw)
    wheel_rgb = complex_to_rgb(Zw, gamma=gamma)

    alpha = np.ones_like(Rw)
    alpha[Rw > 1] = 0.0
    return np.dstack([wheel_rgb, alpha])


def make_neel_color_wheel_rgba(phi_cmap='twilight', gamma=1.0, Nw=200):
    u = np.linspace(-1, 1, Nw)
    v = np.linspace(-1, 1, Nw)
    U, V = np.meshgrid(u, v)

    Rw = np.sqrt(U**2 + V**2)
    Phiw = np.arctan2(V, U)

    magnitude = np.clip(Rw, 0, 1)
    magnitude = magnitude**gamma

    phi_norm = np.mod(Phiw, np.pi) / np.pi
    phi_rgb = plt.get_cmap(phi_cmap)(phi_norm)[..., :3]
    wheel_rgb = phi_rgb * magnitude[..., np.newaxis]

    alpha = np.ones_like(Rw)
    alpha[Rw > 1] = 0.0
    return np.dstack([wheel_rgb, alpha])

__all__ = [
    'plot_some_diffraction_patterns',
    'make_vector_color_map',
    'plot_probe_maps',
    'plot_theta_phi_maps',
    'plot_scan_positions',
    'create_live_plotter',
]


def plot_some_diffraction_patterns(I_sim, positions, scan_indices, probe_numbers):
    '''Plot a grid of diffraction patterns for specified probe numbers and scan indices.

    I_sim: Simulated intensity patterns, shape (N_probes, N_positions, H, W)

    positions: Scan positions in lab coordinates, shape (N_positions, 2)

    scan_indices: List of indices into the scan positions to plot (e.g. [1, 2, 20, 24])
    
    probe_numbers: List of probe numbers to plot (e.g. [0, 1, 2, 3])
    '''
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, probe_number in enumerate(probe_numbers):      # rows
        for j, scan_idx in enumerate(scan_indices):       # columns
            
            axes[i, j].imshow(
                np.log10(I_sim[probe_number, scan_idx].cpu()+1e-8),
                cmap='inferno'
            )
            
            '''
            axes[i, j].imshow(
                np.log10(I_sim[probe_number, scan_idx].cpu()+1e-8),
                cmap='inferno',
            )
            '''
            axes[i, j].set_title(f'Probe {probe_number}, Pos {scan_idx}, x={positions[scan_idx, 0].item():.2f}, y={positions[scan_idx, 1].item():.2f}')
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

def make_vector_color_map(plot=False):
    '''Create a custom colormap for visualizing vector fields - this is for plotting the phi map
    It is designed so that 0 and 180 degrees map to the same color.'''
    #wheel_colors = "#DA314E",'#2E2E2E',"#4EB955",'#2E2E2E',"#3354A4",'#2E2E2E',"#DA314E"
    #https://meyerweb.com/eric/tools/color-blend/#3354A4:DA314E:1:hex
    #wheel_colors = "#DA314E",'#947552',"#4EB955",'#41877D',"#3354A4",'#874379',"#DA314E"
    #wheel_colors = "#DA314E",'#2E2E2E',"#4EB955",'#2E2E2E',"#3354A4",'#2E2E2E',"#DA314E"
    #wheel_colors = '#2E2E2E',"#3354A4",'#2E2E2E',"#DA314E",'#2E2E2E',"#4EB955",'#2E2E2E' #Jack's strain paper
    #                   A              B        C           D           E           F           G
    wheel_colors = '#41877D',"#3354A4",'#874379',"#DA314E",'#947552',"#4EB955",'#41877D' #Jack's strain paper new colour scheme

    wheel_colors = "#3354A4","#000000","#DA314E","#000000","#4EB955","#000000","#3354A4"

    wheel_colors = "#3354A4",'#874379',"#DA314E",'#947552',"#4EB955",'#41877D',"#3354A4"
    to_rgb = mcolors.ColorConverter().to_rgb
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, color in enumerate(wheel_colors):
        r, g, b = to_rgb(color)
        position = ((i) / 6)
        cdict['red'].append([position, r, r])
        cdict['green'].append([position, g, g])
        cdict['blue'].append([position, b, b])
    RGB_scale = mcolors.LinearSegmentedColormap('CustomMap', cdict)
    RGB_scale.set_bad(color='grey')
    gradient = np.linspace(0, 180, 180)
    gradient = np.vstack((gradient, gradient))
    if plot:
        plt.imshow(gradient, aspect='auto', cmap=RGB_scale)
    return RGB_scale


def plot_probe_maps(probe_amplitude, Lx, Ly):
    """
    Plot complex probe as RGB (phase as hue, magnitude as brightness).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    probe_amplitude_np = _to_numpy(probe_amplitude)
    
    # Convert complex probe to RGB
    probe_rgb = complex_to_rgb(probe_amplitude_np)
    
    ax.imshow(probe_rgb, extent=[-Lx, Lx, -Ly, Ly], origin='lower')
    ax.set_title('Probe (Complex as RGB)')
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    
    # Add color wheel inset
    axins = inset_axes(
        ax,
        width="25%",
        height="25%",
        loc='upper right',
        borderpad=1
    )
    
    # Create color wheel
    Nw = 200
    u = np.linspace(-1, 1, Nw)
    v = np.linspace(-1, 1, Nw)
    U, V = np.meshgrid(u, v)
    
    Rw = np.sqrt(U**2 + V**2)
    Phiw = np.arctan2(V, U)
    
    magnitude = 1 - Rw
    magnitude = np.clip(magnitude, 0, 1)
    
    Zw = magnitude * np.exp(1j * Phiw)
    wheel_rgb = complex_to_rgb(Zw, gamma=1.0)
    
    alpha = np.ones_like(Rw)
    alpha[Rw > 1] = 0.0
    wheel_rgba = np.dstack([wheel_rgb, alpha])
    
    axins.imshow(wheel_rgba, origin='lower', extent=[-1, 1, -1, 1])
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_facecolor((0, 0, 0, 0))
    
    for spine in axins.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.show()



def plot_theta_phi_maps(theta, phi, Lx, Ly,
                       positions=None,
                       theta_cmap='magma',
                       phi_cmap='twilight',
                       dx=0.0, dy=0.0,
                       show_positions=True,
                       label_positions=True,
                       label_axes=True):
    """
    Plot theta and phi heatmaps, optionally overlaying scan positions.
    Parameters:
    - theta: 2D array of theta values (shape HxW)
    - phi: 2D array of phi values (shape HxW)
    - Lx, Ly: Real dimensions of the sample
    - positions: Array of scan positions
    - theta_cmap: Colormap for theta heatmap
    - phi_cmap: Colormap for phi heatmap
    - dx, dy: Offset for position labels
    - show_positions: Whether to show scan positions
    - label_positions: Whether to label scan positions with an index number
    - label_axes: Whether to label axes
    """

    # --- Convert inputs to numpy ---
    theta_np = _to_numpy(theta)
    phi_np   = _to_numpy(phi)

    if positions is not None:
        pos = _to_numpy(positions[:, :2])
    else:
        pos = None

    # Default phi colormap fallback
    if phi_cmap is None:
        phi_cmap = 'hsv'
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Theta heatmap ---
    theta_plot = np.abs(np.cos(theta_np))

    im1 = axes[0].imshow(
        theta_plot,
        extent=[-Lx, Lx, -Ly, Ly],
        origin='lower',
        cmap=theta_cmap,
        vmin=0,
        vmax=1
    )
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label(r'$|l_z|$')
    axes[0].set_title(r'$l_z$ heatmap')

    # --- Overlay positions (theta panel) ---
    if show_positions and pos is not None:
        axes[0].scatter(pos[:, 1], pos[:, 0],
                        c='cyan', marker='x', s=12,
                        label='Scan positions')

        if label_positions:
            for k, (dy_pos, dx_pos) in enumerate(pos):
                axes[0].text(dx_pos - dx, dy_pos - dy, str(k),
                             color='white', fontsize=8,
                             ha='center', va='top')

        axes[0].legend(loc='upper right')

    # --- Phi heatmap ---
    phi_deg = np.rad2deg(phi_np)
    phi_plot = np.mod(phi_deg, 180)

    im2 = axes[1].imshow(
        phi_plot,
        extent=[-Lx, Lx, -Ly, Ly],
        origin='lower',
        cmap=phi_cmap,
        vmin=0,
        vmax=180
    )
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label(r'$\phi$ [deg]')
    cbar2.set_ticks([0, 30, 60, 90, 120, 150, 180])
    axes[1].set_title(r'$\phi$ heatmap')

    if label_axes:
        for ax in axes:
            ax.set_xticks(np.linspace(-Lx, Lx, 5))
            ax.set_yticks(np.linspace(-Ly, Ly, 5))
            ax.set_xlabel(r'$x$ [m]')
            ax.set_ylabel(r'$y$ [m]')
    else:
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

    # --- Overlay positions (phi panel) ---
    if show_positions and pos is not None:
        axes[1].scatter(pos[:, 1], pos[:, 0],
                        c='cyan', marker='x', s=12,
                        label='Scan positions')

        if label_positions:
            for k, (dy_pos, dx_pos) in enumerate(pos):
                axes[1].text(dx_pos - dx, dy_pos - dy, str(k),
                             color='white', fontsize=8,
                             ha='right', va='top')

        axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_scan_positions(scan):
    '''Check how the scan positions look after adding the shifts.'''
    plt.figure(figsize=(6,6))
    positions = scan.positions.detach().cpu().numpy()  # Shape should be (N_probes, N_positions, 2)
    for i in range(positions.shape[0]):
        plt.scatter(positions[i,:,1], positions[i,:,0], label=f'Probe {i} shifts')
    plt.xlabel('X shift (lab coordinates)')
    plt.ylabel('Y shift (lab coordinates)')
    plt.title('Probe positions with shifts')
    plt.legend()
    print('Positions after adding shifts:', scan.positions.shape)  # Should be (N_probes, N_positions, 2)

def create_live_plotter(Lx, Ly,
                       positions=None,
                       theta_cmap='magma',
                       phi_cmap='twilight',
                       dx=0.0, dy=0.0,
                       label_axes=True):
    
    '''
    Generates a plotting function that can be called with new probe amplitude, theta, phi, and loss values to update the visualisation in real-time.

    Use by creating an empty plot:
    plot_update = create_live_plotter(Lx, Ly)
    Then update it after every few ptycho iterations:
    plot_update(probe, theta, phi, loss)

    To save the current figure, pass a filename when calling the updater:
    plot_update(probe, theta, phi, loss, save_filename='my_figure.png')

    '''

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    ax_probe = axes[0]  # Complex probe as RGB
    ax_neel  = axes[1]
    ax_loss  = axes[2]
    ax_positions = axes[3]

    # --- Initial dummy data for probe RGB ---
    dummy_rgb = np.zeros((10, 10, 3))

    im_probe = ax_probe.imshow(dummy_rgb, extent=[-Lx, Lx, -Ly, Ly],
                               origin='lower')
    ax_probe.set_title("Probe (Complex as RGB)")
    ax_probe.set_aspect('equal', adjustable='box')

    im_neel = ax_neel.imshow(dummy_rgb, extent=[-Lx, Lx, -Ly, Ly],
                             origin='lower')
    ax_neel.set_title(r"N\'{e}el direction ($\phi$ cmap, $|L_z|$ brightness)")
    ax_neel.set_aspect('equal', adjustable='box')

    loss_line, = ax_loss.plot([], [])
    ax_loss.set_title("Loss vs iteration")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_yscale('log')

    ax_positions.set_xlabel('X shift (lab coordinates)')
    ax_positions.set_ylabel('Y shift (lab coordinates)')
    ax_positions.set_title('Probe positions with shifts')

    loss_history = []

    display_handle = display(fig, display_id=True)

    # --- Optional positions ---
    if positions is not None:
        pos = np.asarray(positions[:, :2])
    else:
        pos = None

    # --- Add color wheel inset to probe plot ---
    axins = inset_axes(
        ax_probe,
        width="25%",
        height="25%",
        loc='upper right',
        borderpad=1
    )
    
    wheel_rgba = make_color_wheel_rgba(gamma=1.0)
    axins.imshow(wheel_rgba, origin='lower', extent=[-1, 1, -1, 1])
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_facecolor((0, 0, 0, 0))
    
    for spine in axins.spines.values():
        spine.set_visible(False)

    ax_neelins = inset_axes(
        ax_neel,
        width="25%",
        height="25%",
        loc='upper right',
        borderpad=1
    )

    ax_neelins.imshow(make_neel_color_wheel_rgba(phi_cmap=phi_cmap, gamma=1.0), origin='lower', extent=[-1, 1, -1, 1])
    ax_neelins.set_xticks([])
    ax_neelins.set_yticks([])
    ax_neelins.set_facecolor((0, 0, 0, 0))

    for spine in ax_neelins.spines.values():
        spine.set_visible(False)

    def update(probe_amplitude, theta, phi, loss, scan, scan_ref=None, save_filename=None):
        '''Update the plots with new data. Call this function after each iteration of the ptychography reconstruction.'''

        probe_amplitude_np = _to_numpy(probe_amplitude)
        theta_np = _to_numpy(theta)
        phi_np   = _to_numpy(phi)

        # --- Probe as complex RGB ---
        probe_rgb = complex_to_rgb(probe_amplitude_np)
        im_probe.set_data(probe_rgb)
        
        # --- Néel direction ---
        neel_rgb = neel_direction_to_rgb(theta_np, phi_np, phi_cmap=phi_cmap)
        im_neel.set_data(neel_rgb)

        # --- Loss ---
        loss_history.append(_to_numpy(loss))
        loss_line.set_data(range(len(loss_history)), loss_history)
        ax_loss.relim()
        ax_loss.autoscale_view()

        # --- Positions ---
        positions = scan.positions.detach().cpu().numpy()  # Shape should be (N_probes, N_positions, 2)

        # Clear the plot and redraw
        ax_positions.cla()
        # Make a colour map, one colour per probe
        cmap = plt.cm.get_cmap('tab10')
        colours = [cmap(i) for i in range(positions.shape[0])]
        for i in range(positions.shape[0]):
            ax_positions.scatter(positions[i,:,1], positions[i,:,0], label=f'Probe {i} shifts', marker='o', c=colours[i])

        if scan_ref is not None:
            ref_positions = scan_ref.positions.detach().cpu().numpy()
            for i in range(ref_positions.shape[0]):
                ax_positions.scatter(ref_positions[i,:, 1], ref_positions[i,:, 0], c=colours[i], marker='x', s=30)
        ax_positions.set_xlabel('X shift (lab coordinates)')
        ax_positions.set_ylabel('Y shift (lab coordinates)')
        ax_positions.set_title('Probe positions with shifts')
        ax_positions.set_aspect('equal', adjustable='box')
        ax_positions.legend(loc='upper right')

        if save_filename is not None:
            fig.savefig(save_filename, bbox_inches='tight')

        display_handle.update(fig)

    return update
