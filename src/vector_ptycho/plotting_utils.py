import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import os

from vector_ptycho._shared import _to_numpy
from vector_ptycho.Neel_field_sim_utils import *

# Avoid circular import by importing reconstruction_utils only when needed

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
    Plot probe abs and phase.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    probe_amplitude_np = _to_numpy(probe_amplitude)
    im1 = axes[0].imshow(np.abs(probe_amplitude_np), extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap='magma')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label(r'$A$')
    axes[0].set_title('Probe Amplitude')

    im2 = axes[1].imshow(np.angle(probe_amplitude_np), extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label(r'$\psi$ [rad]')
    axes[1].set_title('Probe Phase')
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

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    ax_amp   = axes[0]
    ax_phase = axes[1]
    ax_theta = axes[2]
    ax_phi   = axes[3]
    ax_loss  = axes[4]
    ax_positions = axes[5]

    # Hide unused 6th subplot
    axes[5].axis('off')

    # --- Initial dummy data ---
    dummy = np.zeros((10, 10))

    im_amp = ax_amp.imshow(dummy, extent=[-Lx, Lx, -Ly, Ly],
                           origin='lower', cmap='magma')
    cbar_amp = fig.colorbar(im_amp, ax=ax_amp)
    ax_amp.set_title("Probe Amplitude")

    im_phase = ax_phase.imshow(dummy, extent=[-Lx, Lx, -Ly, Ly],
                               origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    cbar_phase = fig.colorbar(im_phase, ax=ax_phase)
    ax_phase.set_title("Probe Phase")

    im_theta = ax_theta.imshow(dummy, extent=[-Lx, Lx, -Ly, Ly],
                               origin='lower', cmap=theta_cmap,
                               vmin=0, vmax=1)
    cbar_theta = fig.colorbar(im_theta, ax=ax_theta)
    ax_theta.set_title(r"$l_z$ heatmap")

    im_phi = ax_phi.imshow(dummy, extent=[-Lx, Lx, -Ly, Ly],
                           origin='lower', cmap=phi_cmap,
                           vmin=0, vmax=180)
    cbar_phi = fig.colorbar(im_phi, ax=ax_phi)
    cbar_phi.set_ticks([0, 30, 60, 90, 120, 150, 180])
    ax_phi.set_title(r"$\phi$ heatmap")

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

    def update(probe_amplitude, theta, phi, loss, scan, scan_ref=None, save_filename=None):
        '''Update the plots with new data. Call this function after each iteration of the ptychography reconstruction.'''

        probe_amplitude_np = _to_numpy(probe_amplitude)
        theta_np = _to_numpy(theta)
        phi_np   = _to_numpy(phi)


        # --- Probe ---
        im_amp.set_data(np.abs(probe_amplitude_np))
        im_phase.set_data(np.angle(probe_amplitude_np))
        im_amp.set_clim(vmin=np.min(np.abs(probe_amplitude_np)),vmax=np.max(np.abs(probe_amplitude_np)))
        im_phase.set_clim(vmin=-np.pi, vmax=np.pi)
        
        # --- Theta ---
        theta_plot = np.abs(np.cos(theta_np))
        im_theta.set_data(theta_plot)

        # --- Phi ---
        phi_deg = np.rad2deg(phi_np)
        phi_plot = np.mod(phi_deg, 180)
        im_phi.set_data(phi_plot)

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
        ax_positions.legend(loc='upper right')

        if save_filename is not None:
            fig.savefig(save_filename, bbox_inches='tight')

        display_handle.update(fig)

    return update
