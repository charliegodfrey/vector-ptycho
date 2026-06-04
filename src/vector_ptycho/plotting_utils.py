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
    """Map a Néel-vector direction to RGB.

    Parameters
    ----------
    theta : array_like
        Polar angle of the magnetization direction.
    phi : array_like
        In-plane angle of the magnetization direction.
    gamma : float, optional
        Gamma correction applied to the brightness channel.
    phi_cmap : str, optional
        Matplotlib colormap used to encode phi with pi-periodic wrapping.

    Returns
    -------
    numpy.ndarray
        RGB image where hue comes from phi and brightness fades with
        |Lz| = |cos(theta)|.
    """
    lz = np.abs(np.cos(theta))
    brightness = np.clip(1.0 - lz, 0.0, 1.0)
    brightness = brightness**gamma

    phi_norm = np.mod(phi, np.pi) / np.pi
    phi_rgb = plt.get_cmap(phi_cmap)(phi_norm)[..., :3]
    return phi_rgb * brightness[..., np.newaxis]


def make_color_wheel_rgba(gamma=1.0, Nw=200):
    """Build a complex-probe color wheel with transparent background.

    The wheel is used as an inset legend for the complex-valued probe plots.
    Phase is encoded as hue and magnitude as brightness.
    """
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
    """Build a Néel-direction color wheel with pi-periodic phi wrapping.

    The wheel matches the Néel visualization: phi is mapped through the
    supplied colormap, the center is black, and the outer ring is fully
    saturated.
    """
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


def _decorate_probe_color_wheel_axes(ax, label_color='white'):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor((0, 0, 0, 0))

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.axhline(0, color=label_color, linewidth=0.8, alpha=0.75)
    ax.axvline(0, color=label_color, linewidth=0.8, alpha=0.75)

    text_kwargs = dict(color=label_color, fontsize=12, ha='center', va='center')
    ax.text(1.18, 0.0, r'$0$', **text_kwargs)
    ax.text(0.0, 1.18, r'$\pi/2$', **text_kwargs)
    ax.text(-1.18, 0.0, r'$\pi$', **text_kwargs)
    ax.text(0.0, -1.18, r'$3\pi/2$', **text_kwargs)

__all__ = [
    'plot_some_diffraction_patterns',
    'make_vector_color_map',
    'plot_probe_maps',
    'plot_theta_phi_maps',
    'plot_scan_positions',
    'create_live_plotter',
    'load_checkpoint_history',
    'plot_checkpoint_history',
]


def load_checkpoint_history(checkpoint_files):
    """Concatenate iteration and cosine-similarity histories across checkpoints.

    Each checkpoint is treated as a continuation of the previous one, so the
    returned iteration numbers are shifted to keep the x-axis continuous.
    """
    combined_iterations = []
    combined_cosine_similarity = []
    iteration_offset = 0

    for checkpoint_file in checkpoint_files:
        ckpt = torch.load(checkpoint_file, map_location='cpu')
        iteration_numbers = np.asarray(ckpt['iteration_numbers'])
        cosine_similarity_history = np.asarray(ckpt['cosine_similarity_history'])

        if iteration_numbers.size == 0:
            continue

        if combined_iterations:
            iteration_numbers = iteration_numbers - iteration_numbers[0] + iteration_offset

        combined_iterations.append(iteration_numbers)
        combined_cosine_similarity.append(cosine_similarity_history)
        iteration_offset = iteration_numbers[-1] + 1

    if not combined_iterations:
        return np.array([]), np.array([])

    return np.concatenate(combined_iterations), np.concatenate(combined_cosine_similarity)


def plot_checkpoint_history(checkpoint_files, label=None, ax=None, line_styles=None, colour=None, **plot_kwargs):
    """Plot cosine similarity history for a sequence of checkpoint files.

    Each checkpoint file is drawn as its own line segment so you can give each
    file a different linestyle while keeping one legend entry per run series.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    iteration_offset = 0

    for i, checkpoint_file in enumerate(checkpoint_files):
        ckpt = torch.load(checkpoint_file, map_location='cpu')
        iteration_numbers = np.asarray(ckpt['iteration_numbers'])
        cosine_similarity_history = np.asarray(ckpt['cosine_similarity_history'])

        if iteration_numbers.size == 0:
            continue

        # Keep the x-axis continuous across checkpoint boundaries.
        if i > 0:
            iteration_numbers = iteration_numbers - iteration_numbers[0] + iteration_offset

        if line_styles:
            style = line_styles[i % len(line_styles)]
        else:
            style = '-'

        ax.plot(
            iteration_numbers,
            cosine_similarity_history,
            label=label if i == 0 else None,
            linestyle=style,
            color=colour,
            **plot_kwargs,
        )

        iteration_offset = iteration_numbers[-1] + 1

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cosine Similarity')
    #ax.set_title('Ptycho Reconstruction Progress')
    return ax


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
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    probe_amplitude_np = _to_numpy(probe_amplitude)
    
    # Convert complex probe to RGB
    probe_rgb = complex_to_rgb(probe_amplitude_np)
    
    ax.imshow(probe_rgb, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], origin='lower')
    ax.set_title('Probe (Complex as RGB)')
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    
    # Add color wheel inset
    axins = inset_axes(
        ax,
        width="18%",
        height="18%",
        loc='upper right',
        borderpad=2
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
    _decorate_probe_color_wheel_axes(axins)
    
    plt.tight_layout()
    plt.show()


def _extract_probe_inset_view(probe_rgb, Lx, Ly, probe_inset_crop=None):
    """Return a cropped probe image and matching extent for an inset."""
    probe_rgb_np = np.asarray(probe_rgb)
    if probe_rgb_np.ndim < 2:
        return probe_rgb_np, [-Lx/2, Lx/2, -Ly/2, Ly/2]

    height, width = probe_rgb_np.shape[:2]

    if probe_inset_crop is None:
        row_start, row_end = 0, height
        col_start, col_end = 0, width
    elif isinstance(probe_inset_crop, int):
        crop_height = min(int(probe_inset_crop), height)
        crop_width = min(int(probe_inset_crop), width)
        row_start = max((height - crop_height) // 2, 0)
        col_start = max((width - crop_width) // 2, 0)
        row_end = row_start + crop_height
        col_end = col_start + crop_width
    elif len(probe_inset_crop) == 2:
        crop_height = min(int(probe_inset_crop[0]), height)
        crop_width = min(int(probe_inset_crop[1]), width)
        row_start = max((height - crop_height) // 2, 0)
        col_start = max((width - crop_width) // 2, 0)
        row_end = row_start + crop_height
        col_end = col_start + crop_width
    elif len(probe_inset_crop) == 4:
        row_start, row_end, col_start, col_end = (int(value) for value in probe_inset_crop)
        row_start = max(row_start, 0)
        col_start = max(col_start, 0)
        row_end = min(max(row_end, row_start + 1), height)
        col_end = min(max(col_end, col_start + 1), width)
    else:
        raise ValueError(
            "probe_inset_crop must be None, an int, a (height, width) tuple, or a (row_start, row_end, col_start, col_end) tuple"
        )

    y_edges = np.linspace(-Ly, Ly, height + 1)
    x_edges = np.linspace(-Lx, Lx, width + 1)
    extent = [x_edges[col_start], x_edges[col_end], y_edges[row_start], y_edges[row_end]]
    return probe_rgb_np[row_start:row_end, col_start:col_end], extent



def plot_theta_phi_maps(theta, phi, Lx, Ly,
                       positions=None,
                       theta_cmap='magma',
                       phi_cmap='twilight',
                       dx=0.0, dy=0.0,
                       show_positions=True,
                       label_positions=True,
                       label_axes=True,
                       probe_amplitude=None,
                       show_probe_inset=False,
                       probe_inset_crop=None,
                       title_on=False):
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
        - probe_amplitude: Optional complex probe to show as a cropped inset
        - show_probe_inset: Whether to add the probe inset in the lower right
        - probe_inset_crop: Optional crop for the probe inset. Use an int for a
            centered square crop, a (height, width) tuple for a centered rectangle,
            or (row_start, row_end, col_start, col_end) for explicit bounds.
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
 
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    # --- Theta heatmap ---
    theta_plot = np.abs(np.cos(theta_np))

    # --- Phi visualization: colour = phi (pi-periodic), brightness = |L_z| ---
    neel_rgb = neel_direction_to_rgb(theta_np, phi_np, phi_cmap=phi_cmap)

    im = axes.imshow(
        neel_rgb,
        extent=[-Lx, Lx, -Ly, Ly],
        origin='lower'
    )

    # Add Neel color wheel inset to explain the mapping
    ax_neelins_local = inset_axes(
        axes,
        width="18%",
        height="18%",
        loc='upper right',
        borderpad=2
    )
    ax_neelins_local.imshow(make_neel_color_wheel_rgba(phi_cmap=phi_cmap, gamma=1.0), origin='lower', extent=[-1, 1, -1, 1])
    _decorate_probe_color_wheel_axes(ax_neelins_local)

    if show_probe_inset and probe_amplitude is not None:
        probe_amplitude_np = _to_numpy(probe_amplitude)
        probe_rgb = complex_to_rgb(probe_amplitude_np)
        probe_inset_rgb, probe_inset_extent = _extract_probe_inset_view(
            probe_rgb,
            Lx,
            Ly,
            probe_inset_crop=probe_inset_crop,
        )

        ax_probeins_local = inset_axes(
            axes,
            width="28%",
            height="28%",
            loc='lower right',
            borderpad=1.2,
        )
        ax_probeins_local.imshow(
            probe_inset_rgb,
            origin='lower',
            extent=probe_inset_extent,
        )
        ax_probeins_local.set_xticks([])
        ax_probeins_local.set_yticks([])
        ax_probeins_local.set_facecolor((0, 0, 0, 0))
        for spine in ax_probeins_local.spines.values():
            spine.set_visible(False)
    if title_on:
        axes.set_title(r'$\phi$ heatmap (colour: $\phi$, brightness: $|L_z|$)')

    if label_axes:
        axes.set_xlabel(r'$x$ [m]')
        axes.set_ylabel(r'$y$ [m]')
    else:
        axes.set_xticks([])
        axes.set_yticks([])

    # --- Overlay positions (phi panel) ---
    if show_positions and pos is not None:
        axes.scatter(pos[:, 1], pos[:, 0],
                        c='cyan', marker='x', s=12,
                        label='Scan positions')

        if label_positions:
            for k, (dy_pos, dx_pos) in enumerate(pos):
                axes.text(dx_pos - dx, dy_pos - dy, str(k),
                             color='white', fontsize=8,
                             ha='right', va='top')

        axes.legend(loc='upper left')

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
    Generates a plotting function that can be called with new probe amplitude,
    theta, phi, loss, and optional diffraction data to update the
    visualisation in real-time.

    Use by creating an empty plot:
    plot_update = create_live_plotter(Lx, Ly)
    Then update it after every few ptycho iterations:
    plot_update(probe, theta, phi, loss)

    To save the current figure, pass a filename when calling the updater:
    plot_update(probe, theta, phi, loss, save_filename='my_figure.png')

    '''

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    axes = axes.ravel()

    ax_probe = axes[0]  # Complex probe as RGB
    ax_neel  = axes[1]
    ax_loss  = axes[2]
    ax_positions = axes[3]
    ax_diff_sim = axes[4]
    ax_diff_exp = axes[5]

    # --- Initial dummy data for probe RGB ---
    dummy_rgb = np.zeros((10, 10, 3))

    im_probe = ax_probe.imshow(dummy_rgb, extent=[-Lx, Lx, -Ly, Ly],
                               origin='lower')
    ax_probe.set_title("Probe (Complex as RGB)")
    ax_probe.set_aspect('equal', adjustable='box')

    im_neel = ax_neel.imshow(dummy_rgb, extent=[-Lx, Lx, -Ly, Ly],
                             origin='lower')
    ax_neel.set_title(r"Néel direction ($\phi$ cmap, $|L_z|$ brightness)")
    ax_neel.set_aspect('equal', adjustable='box')

    dummy_diff = np.zeros((10, 10))
    im_diff_sim = ax_diff_sim.imshow(dummy_diff, origin='lower', cmap='inferno')
    ax_diff_sim.set_title('Simulated diffraction (log10)')
    ax_diff_sim.set_xlabel('Detector x [px]')
    ax_diff_sim.set_ylabel('Detector y [px]')

    im_diff_exp = ax_diff_exp.imshow(dummy_diff, origin='lower', cmap='inferno')
    ax_diff_exp.set_title('Experimental diffraction (log10)')
    ax_diff_exp.set_xlabel('Detector x [px]')
    ax_diff_exp.set_ylabel('Detector y [px]')

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
        width="18%",
        height="18%",
        loc='upper right',
        borderpad=2
    )
    
    wheel_rgba = make_color_wheel_rgba(gamma=1.0)
    axins.imshow(wheel_rgba, origin='lower', extent=[-1, 1, -1, 1])
    _decorate_probe_color_wheel_axes(axins)

    ax_neelins = inset_axes(
        ax_neel,
        width="18%",
        height="18%",
        loc='upper right',
        borderpad=2
    )

    ax_neelins.imshow(make_neel_color_wheel_rgba(phi_cmap=phi_cmap, gamma=1.0), origin='lower', extent=[-1, 1, -1, 1])
    _decorate_probe_color_wheel_axes(ax_neelins)

    def _extract_diffraction_frame(intensity, probe_idx=0, scan_idx=0):
        """Extract a single detector frame from 2D/3D/4D intensity arrays."""
        if intensity is None:
            return None

        arr = _to_numpy(intensity)
        if arr.ndim == 4:
            p = int(np.clip(probe_idx, 0, arr.shape[0] - 1))
            s = int(np.clip(scan_idx, 0, arr.shape[1] - 1))
            return arr[p, s], p, s
        if arr.ndim == 3:
            s = int(np.clip(scan_idx, 0, arr.shape[0] - 1))
            return arr[s], None, s
        if arr.ndim == 2:
            return arr, None, None
        return None

    def update(
        probe_amplitude,
        theta,
        phi,
        loss,
        scan,
        scan_ref=None,
        I_sim=None,
        I_exp=None,
        diff_probe_idx=0,
        diff_scan_idx=0,
        save_filename=None,
    ):
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

        # --- Diffraction patterns (shared colour scale between sim/exp) ---
        sim_frame_info = _extract_diffraction_frame(I_sim, diff_probe_idx, diff_scan_idx)
        sim_plot = None
        if sim_frame_info is not None:
            sim_frame, p_idx, s_idx = sim_frame_info
            sim_plot = np.log10(np.clip(sim_frame, 0.0, None) + 1e-8)
            im_diff_sim.set_data(sim_plot)
            if p_idx is None and s_idx is None:
                ax_diff_sim.set_title('Simulated diffraction (log10)')
            elif p_idx is None:
                ax_diff_sim.set_title(f'Simulated diffraction (scan {s_idx}, log10)')
            else:
                ax_diff_sim.set_title(f'Simulated diffraction (probe {p_idx}, scan {s_idx}, log10)')

        exp_frame_info = _extract_diffraction_frame(I_exp, diff_probe_idx, diff_scan_idx)
        exp_plot = None
        if exp_frame_info is not None:
            exp_frame, p_idx, s_idx = exp_frame_info
            exp_plot = np.log10(np.clip(exp_frame, 0.0, None) + 1e-8)
            im_diff_exp.set_data(exp_plot)
            if p_idx is None and s_idx is None:
                ax_diff_exp.set_title('Experimental diffraction (log10)')
            elif p_idx is None:
                ax_diff_exp.set_title(f'Experimental diffraction (scan {s_idx}, log10)')
            else:
                ax_diff_exp.set_title(f'Experimental diffraction (probe {p_idx}, scan {s_idx}, log10)')

        available_plots = [arr for arr in (sim_plot, exp_plot) if arr is not None]
        if available_plots:
            global_vmin = float(min(np.min(arr) for arr in available_plots))
            global_vmax = float(max(np.max(arr) for arr in available_plots))
            if global_vmax <= global_vmin:
                global_vmax = global_vmin + 1e-12

            if sim_plot is not None:
                im_diff_sim.set_clim(vmin=global_vmin, vmax=global_vmax)
            if exp_plot is not None:
                im_diff_exp.set_clim(vmin=global_vmin, vmax=global_vmax)

        if save_filename is not None:
            fig.savefig(save_filename, bbox_inches='tight')

        display_handle.update(fig)

    return update
