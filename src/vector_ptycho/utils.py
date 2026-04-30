import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import os

def make_vector_color_map(plot=False):
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

def _to_numpy(x):
    """Safely convert torch / array-like to numpy."""
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)

def plot_probe_maps(probe_amplitude, Lx, Ly):
    """
    Plot probe abs and phase.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    im1 = axes[0].imshow(np.abs(probe_amplitude), extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap='magma')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label(r'$A$')
    axes[0].set_title('Probe Amplitude')

    im2 = axes[1].imshow(np.angle(probe_amplitude), extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap='twilight')
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




def create_live_plotter(Lx, Ly,
                       positions=None,
                       theta_cmap='magma',
                       phi_cmap='twilight',
                       dx=0.0, dy=0.0,
                       show_positions=False,
                       label_positions=False,
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

    # Hide unused 6th subplot
    axes[5].axis('off')

    # --- Initial dummy data ---
    dummy = np.zeros((10, 10))

    im_amp = ax_amp.imshow(dummy, extent=[-Lx, Lx, -Ly, Ly],
                           origin='lower', cmap='magma')
    cbar_amp = fig.colorbar(im_amp, ax=ax_amp)
    ax_amp.set_title("Probe Amplitude")

    im_phase = ax_phase.imshow(dummy, extent=[-Lx, Lx, -Ly, Ly],
                               origin='lower', cmap='twilight')
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
    
    loss_history = []

    display_handle = display(fig, display_id=True)

    # --- Optional positions ---
    if positions is not None:
        pos = np.asarray(positions[:, :2])
    else:
        pos = None

    def update(probe_amplitude, theta, phi, loss, save_filename=None):
        probe_amplitude_np = _to_numpy(probe_amplitude)
        theta_np = _to_numpy(theta)
        phi_np   = _to_numpy(phi)


        # --- Probe ---
        im_amp.set_data(np.abs(probe_amplitude_np))
        im_phase.set_data(np.angle(probe_amplitude_np))
        im_amp.set_clim(vmin=np.min(np.abs(probe_amplitude_np)),vmax=np.max(np.abs(probe_amplitude_np)))
        im_phase.set_clim(vmin=np.min(np.angle(probe_amplitude_np)),vmax=np.max(np.angle(probe_amplitude_np)))
        
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

        # --- Overlay positions (only once ideally, but safe here) ---
        if show_positions and pos is not None:
            for ax in [ax_theta, ax_phi]:
                ax.scatter(pos[:, 1], pos[:, 0],
                           c='cyan', marker='x', s=12)

        if save_filename is not None:
            fig.savefig(save_filename, bbox_inches='tight')

        display_handle.update(fig)

    return update

def im_display2(*images, titles=None, limits=None, save=False, name='name', c='viridis', bar=0):
    '''A new version of im_display which allows colour bars and changing the color map and saving'''
    fig, axes = plt.subplots(ncols=len(images))
    
    cmap = matplotlib.colormaps.get_cmap(c)  # viridis is the default colormap for imshow
    cmap.set_bad(color='red')
    
    if not hasattr(axes, '__iter__'):
        axes = (axes, )
    for i, (image, ax) in enumerate(zip(images, axes)):
        if limits==None:
            im = ax.imshow(image, cmap)
        else:
            im = ax.imshow(image, cmap, vmin=np.min(limits), vmax=np.max(limits))
        if (bar == 2):
            cbar = ax.figure.colorbar(im)
        ax.axis('off')
        if titles is not None: 
            ax.set_title(titles[i], size=10)
        #ax.grid(color='red', linestyle='-.', linewidth=1) #Charlie put this here to look at defects in distortion corrected images.
    
    if (bar==1):
        cbar_ax = fig.add_axes([0.92, 0.4, 0.03, 0.15])
        fig.colorbar(im, cax=cbar_ax)
    
    fig.set_size_inches(12, 4)
    
    if (save == True):
        plt.savefig(name, dpi=1000)


device = "cuda" if torch.cuda.is_available() else "cpu"
cdtype = torch.complex64
eps = 1e-8

# =========================
# Fourier operators
# =========================
def F(x):
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.fftshift(x, dim=(-2, -1)), norm='forward'),
        dim=(-2, -1)
    )

def iF(x):
    return torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='backward'),
        dim=(-2, -1)
    )

# =========================
# Core classes
# =========================
class JonesField:
    def __init__(self, Ex, Ey):
        self.Ex = Ex
        self.Ey = Ey


class Probe:
    def __init__(self, amplitude, jones_vector, fluence=None, normalized=False):
        self.amplitude = amplitude
        self.jones_vector = jones_vector
        self.fluence = fluence if fluence is not None else 1.0  # Optional scalar for total probe photons per exposure, used for Poisson noise simulation.
        if normalized:
            self.amplitude_scaled = self.amplitude
        else:
            self.amplitude_scaled = self.amplitude * torch.sqrt(self.fluence / torch.sum(torch.abs(self.amplitude)**2))  # Normalise probe amplitude  
            self.amplitude = self.amplitude_scaled

    def field(self):
        Ex = self.amplitude * self.jones_vector[0]
        Ey = self.amplitude * self.jones_vector[1]
        return JonesField(Ex, Ey)

    def shifted(self, dy, dx):
        amp = torch.roll(self.amplitude, shifts=(dy, dx), dims=(0, 1))
        return Probe(amp, self.jones_vector, fluence=self.fluence, normalized=True)


class JonesObject:
    def __init__(self, J):
        self.J = J  # (H, W, 2, 2)

    def apply(self, field: JonesField):
        Jxx = self.J[...,0,0]
        Jxy = self.J[...,0,1]
        Jyx = self.J[...,1,0]
        Jyy = self.J[...,1,1]

        Ex = Jxx * field.Ex + Jxy * field.Ey
        Ey = Jyx * field.Ex + Jyy * field.Ey

        return JonesField(Ex, Ey)


# =========================
# Néel object parameterisation
# =========================

class NeelObject:
    def __init__(self, C, A1, A2):
        """
        Pure physics object — no optimisation variables.

        C  : complex scalar (charge scattering)
        A1 : complex scalar (in-plane XMLD)
        A2 : complex scalar (out-of-plane XMLD)
        """
        self.C = C
        self.A1 = A1
        self.A2 = A2
    def build_jones_from_cartesian(self, lx, ly, lz):
        """
        Build Jones matrix from Cartesian components.

        lx, ly, lz: real tensors (H, W)
        """
        theta = torch.acos(lz)
        phi = torch.atan2(ly, lx)
        return self.build_jones(theta, phi)
    
    def build_jones(self, theta, phi):
        """
        Build Jones matrix from physical angles.

        theta, phi: real tensors (H, W)
        returns: (H, W, 2, 2) complex
        """

        # Convert to complex for safe algebra
        #theta = theta.to(torch.complex64)
        #phi   = phi.to(torch.complex64)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        cos_p = torch.cos(phi)
        sin_p = torch.sin(phi)

        Jxx = self.C + self.A1 * (cos_p**2 * sin_t**2) + self.A2 * (cos_t**2)
        Jxy = self.A1 * (sin_p * cos_p * sin_t**2)
        Jyx = Jxy
        Jyy = self.C + self.A1 * (sin_p**2 * sin_t**2) + self.A2 * (cos_t**2)

        J = torch.stack([
            torch.stack([Jxx, Jxy], dim=-1),
            torch.stack([Jyx, Jyy], dim=-1)
        ], dim=-2)

        return J


# =========================
# Inverse: Jones → Néel
# =========================
def jones_to_neel(J):
    """
    Extract approximate theta and phi from Jones matrix.
    Assumes linear birefringence model.
    """
    Jxx = J[...,0,0]
    Jxy = J[...,0,1]

    # Orientation estimate
    theta = 0.5 * torch.atan2(
        2 * torch.real(Jxy),
        torch.real(Jxx - J[...,1,1])
    )

    # Retardance estimate
    phi = torch.angle(J[...,1,1])  # approximate

    return theta, phi


# =========================
# Propagation & detection
# =========================
class Propagator:
    def propagate(self, field: JonesField):
        return JonesField(F(field.Ex), F(field.Ey))


class Detector:
    def __init__(self, add_poisson_noise=False):
        self.add_poisson_noise = add_poisson_noise

    def intensity(self, field: JonesField):
        intensity = torch.abs(field.Ex)**2 + torch.abs(field.Ey)**2

        if self.add_poisson_noise:
            intensity = torch.poisson(torch.clamp(intensity, min=0))

        return intensity


# =========================
# Scan trajectory
# =========================
class ScanTrajectory:
    def __init__(self, positions):
        self.positions = torch.tensor(positions, device=device)


# =========================
# Forward model
# =========================
class ForwardModel:
    def __init__(self, obj, propagator, detector):
        self.obj = obj
        self.propagator = propagator
        self.detector = detector

    def simulate_all(self, probes, scan, position_indices=None):
        data = []

        if position_indices is None:
            positions = scan.positions
        else:
            idx = torch.as_tensor(position_indices, dtype=torch.long, device=scan.positions.device)
            positions = scan.positions.index_select(0, idx)

        for probe in probes:
            probe_data = []

            for dy, dx in positions.tolist():
                field = probe.shifted(int(dy), int(dx)).field()
                field = self.obj.apply(field)
                field = self.propagator.propagate(field)
                I = self.detector.intensity(field)

                probe_data.append(I)

            data.append(torch.stack(probe_data, dim=0))

        return torch.stack(data, dim=0)


def make_meron_antimeron_theta_phi(
    Nx=300,
    Ny=300,
    Lx=10.0,
    Ly=None,
    r1=(-2.5, 0.0),
    r2=(2.5, 0.0),
    sigma=0.5,
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
    x = np.linspace(-Lx, Lx, Nx, dtype=np.float32)
    y = np.linspace(-Ly, Ly, Ny, dtype=np.float32)
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

def initialize_probe_amplitude(H, W, Lx, Ly, device):
    '''

    Initialise the probe amplitude with a guess that has a similar shape to the true probe, but is not exactly the same. This is to help convergence of the joint object-probe optimization.
    '''
    x = torch.linspace(-Lx, Lx, H, device=device)
    y = torch.linspace(-Ly, Ly, W, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    R = torch.sqrt(X**2 + Y**2)
    diffuser = torch.rand((H, W), device=device) #0.7 * (torch.sin(1 * R) + torch.cos((Y * 1 + X * 1) - 0.8 * (X - 0.2)) + torch.cos((Y - X) - 0.5 * (X)))
    P = torch.exp(2j * np.pi * diffuser) * (torch.exp(-0.5 * R))
    return P

def estimate_probe(H, W, Lx, Ly, I_meas, device):
    '''
    Estimate the probe intensity by square root of the diffraction pattern averaged over all scan positions. 
    This is a common heuristic for probe initialization in ptychography.
    Then inverse Fourier Transform to get an initial guess for the probe amplitude in real space.
    '''
    I_avg = I_meas.mean(dim=(0, 1))  # Average over scan positions
    print('The shape of the diffraction pattern is: ', I_avg.shape)
    P = torch.sqrt(I_avg)
    P = iF(P)  # Inverse Fourier Transform to get initial probe guess in real space
    return P


class Reconstruction:
    """Joint object-probe reconstruction with checkpoint save/load + resumable run()."""

    def __init__(
        self,
        scan,
        pol_angles,
        I_meas,
        H,
        W,
        Lx,
        Ly,
        fluence,
        RGB_scale,
        device=None,
        initial_theta=None,
        initial_phi=None,
        initial_probe_amplitude=None,
        initial_C=None,
        initial_A1=None,
        initial_A2=None,
        start_iteration=0,
        loss_history=None,
    ):
        if device is None:
            device = I_meas.device

        self.scan = scan
        self.pol_angles = pol_angles
        self.I_meas = I_meas
        self.H = H
        self.W = W
        self.Lx = Lx
        self.Ly = Ly
        self.fluence = fluence
        self.RGB_scale = RGB_scale
        self.device = device

        self.theta = torch.nn.Parameter(initial_theta.clone().to(device))
        self.phi = torch.nn.Parameter(initial_phi.clone().to(device))
        self.probe_amplitude = torch.nn.Parameter(initial_probe_amplitude.clone().to(device))

        self.C = torch.nn.Parameter(initial_C.clone().to(device))
        self.A1 = torch.nn.Parameter(initial_A1.clone().to(device))
        self.A2 = torch.nn.Parameter(initial_A2.clone().to(device))

        self.start_iteration = int(start_iteration)
        self.loss_history = list(loss_history) if loss_history is not None else []
        self.optimizer = None
        self.optimizer_config = None

    @classmethod
    def from_initial_guess(
        cls,
        scan,
        pol_angles,
        I_meas,
        H,
        W,
        Lx,
        Ly,
        fluence,
        RGB_scale,
        C,
        A1,
        A2,
        device=None,
        initial_object='random',
        initial_probe='gaussian',
        initial_probe_amplitude=None,
    ):
        if device is None:
            device = I_meas.device

        if initial_object == 'random_IP_90deg':
            initial_theta = torch.rand((H, W), device=device) * 0.01 + torch.pi / 2
            initial_phi = torch.rand((H, W), device=device) * 0.1 + torch.pi / 2
        if initial_object == 'random':
            initial_theta = torch.rand((H, W), device=device) * 0.1 + torch.pi / 2
            initial_phi = torch.rand((H, W), device=device) * 1.0 + torch.pi / 2
        if initial_probe == 'gaussian':
            initial_probe_amplitude = initialize_probe_amplitude(H, W, Lx, Ly, device)
        if initial_probe == 'correct':
            if initial_probe_amplitude is None:
                raise ValueError(
                    "initial_probe_amplitude must be provided when initial_probe='correct'"
                )
            initial_probe_amplitude = initial_probe_amplitude.to(device)
        if initial_probe not in {'gaussian', 'correct'}:
            raise ValueError("initial_probe must be one of {'gaussian', 'correct'}")

        return cls(
            scan=scan,
            pol_angles=pol_angles,
            I_meas=I_meas,
            H=H,
            W=W,
            Lx=Lx,
            Ly=Ly,
            fluence=fluence,
            RGB_scale=RGB_scale,
            device=device,
            initial_theta=initial_theta,
            initial_phi=initial_phi,
            initial_probe_amplitude=initial_probe_amplitude,
            initial_C=C.detach().clone(),
            initial_A1=A1.detach().clone(),
            initial_A2=A2.detach().clone(),
            start_iteration=0,
            loss_history=[],
        )

    def to_checkpoint_dict(self):
        if torch.is_tensor(self.fluence):
            fluence_to_save = self.fluence.detach().cpu()
        else:
            fluence_to_save = torch.as_tensor(self.fluence)

        return {
            "iteration": int(self.start_iteration),
            "theta": self.theta.detach().cpu(),
            "phi": self.phi.detach().cpu(),
            "probe_amplitude": self.probe_amplitude.detach().cpu(),
            "C": self.C.detach().cpu(),
            "A1": self.A1.detach().cpu(),
            "A2": self.A2.detach().cpu(),
            "scan_positions": self.scan.positions.detach().cpu(),
            "I_meas": self.I_meas.detach().cpu(),
            "fluence": fluence_to_save,
            "loss_history": list(self.loss_history),
            "config": {
                "H": int(self.H),
                "W": int(self.W),
                "Lx": float(self.Lx),
                "Ly": float(self.Ly),
                "pol_angles": list(self.pol_angles),
            },
        }

    def save(self, checkpoint_path):
        torch.save(self.to_checkpoint_dict(), checkpoint_path)

    @classmethod
    def load(
        cls,
        checkpoint_path,
        scan=None,
        pol_angles=None,
        I_meas=None,
        fluence=None,
        RGB_scale=None,
        Lx=None,
        Ly=None,
        device=None,
    ):
        """
        Load a reconstruction checkpoint using only explicit inputs and checkpoint data.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device if device is not None else "cpu")
        config = checkpoint.get("config", {})

        scan_positions_ckpt = checkpoint.get("scan_positions", None)
        I_meas_ckpt = checkpoint.get("I_meas", None)
        fluence_ckpt = checkpoint.get("fluence", None)

        # Resolve optional runtime data from checkpoint first.
        if scan is None:
            if scan_positions_ckpt is not None:
                scan = ScanTrajectory(scan_positions_ckpt.to(torch.int64))
        if pol_angles is None:
            pol_angles = config.get("pol_angles", None)
        if I_meas is None:
            if I_meas_ckpt is not None:
                I_meas = I_meas_ckpt
        if fluence is None:
            if fluence_ckpt is not None:
                fluence = fluence_ckpt
        if RGB_scale is None:
            raise ValueError("RGB_scale must be provided explicitly to Reconstruction.load().")
        if Lx is None:
            Lx = config.get("Lx", None)
        if Ly is None:
            Ly = config.get("Ly", None)

        missing = []
        if scan is None:
            missing.append("scan")
        if pol_angles is None:
            missing.append("pol_angles")
        if I_meas is None:
            missing.append("I_meas")
        if fluence is None:
            missing.append("fluence")
        if RGB_scale is None:
            missing.append("RGB_scale")
        if Lx is None:
            missing.append("Lx")
        if Ly is None:
            missing.append("Ly")

        if missing:
            raise ValueError(f"Missing required arguments/checkpoint fields for load(): {missing}")

        if device is None:
            device = I_meas.device

        I_meas = I_meas.to(device)
        if torch.is_tensor(fluence):
            fluence = fluence.to(device)
        else:
            fluence = torch.as_tensor(fluence, device=device)

        theta = checkpoint["theta"].to(device)
        phi = checkpoint["phi"].to(device)
        probe_amplitude = checkpoint["probe_amplitude"].to(device)
        C = checkpoint["C"].to(device)
        A1 = checkpoint["A1"].to(device)
        A2 = checkpoint["A2"].to(device)

        H, W = int(theta.shape[0]), int(theta.shape[1])

        recon = cls(
            scan=scan,
            pol_angles=pol_angles,
            I_meas=I_meas,
            H=H,
            W=W,
            Lx=Lx,
            Ly=Ly,
            fluence=fluence,
            RGB_scale=RGB_scale,
            device=device,
            initial_theta=theta,
            initial_phi=phi,
            initial_probe_amplitude=probe_amplitude,
            initial_C=C,
            initial_A1=A1,
            initial_A2=A2,
            start_iteration=int(checkpoint.get("iteration", 0)),
            loss_history=list(checkpoint.get("loss_history", [])),
        )
        return recon

    def initialize_optimizer(
        self,
        optimizer_cls=torch.optim.Adam,
        lr_theta=1e-2,
        lr_phi=1e-2,
        lr_object=1e-5,
        lr_probe=1e-2,
        **optimizer_kwargs,
    ):
        """Create and store an optimizer for the current reconstruction parameters."""
        self.optimizer_config = {
            "optimizer_cls": optimizer_cls,
            "lr_theta": lr_theta,
            "lr_phi": lr_phi,
            "lr_object": lr_object,
            "lr_probe": lr_probe,
            **optimizer_kwargs,
        }
        self.optimizer = optimizer_cls([
            {"params": [self.theta], "lr": lr_theta},
            {"params": [self.phi], "lr": lr_phi},
            {"params": [self.C, self.A1, self.A2], "lr": lr_object},
            {"params": [self.probe_amplitude], "lr": lr_probe},
        ], **optimizer_kwargs)
        return self.optimizer

    def run(
        self,
        num_iterations=100,
        batch_size=10,
        checkpoint_path=None,
        checkpoint_every=100,
        optimizer=None,
        optimizer_cls=torch.optim.Adam,
        lr_theta=1e-2,
        lr_phi=1e-2,
        lr_object=1e-5,
        lr_probe=1e-2,
        **optimizer_kwargs,
    ):
        eps = 1e-12
        cdtype = torch.complex64

        probe_amplitude_start = self.probe_amplitude.detach()
        fluence_calc = torch.sum(torch.abs(probe_amplitude_start) ** 2)


        print(
            f"Calculated initial fluence from probe amplitude: {fluence_calc.item():.6e}, "
            f"target fluence: {self.fluence.item():.6e}"
        )

        if optimizer is None:
            if self.optimizer is not None:
                optimizer = self.optimizer
            else:
                optimizer = self.initialize_optimizer(
                    optimizer_cls=optimizer_cls,
                    lr_theta=lr_theta,
                    lr_phi=lr_phi,
                    lr_object=lr_object,
                    lr_probe=lr_probe,
                    **optimizer_kwargs,
                )
        else:
            self.optimizer = optimizer

        n_scan = int(self.scan.positions.shape[0])
        if n_scan < 1:
            raise ValueError("scan.positions must contain at least one position")

        batch_size = int(max(1, min(batch_size, n_scan)))
        steps_per_epoch = (n_scan + batch_size - 1) // batch_size
        scan_device = self.scan.positions.device

        for local_iteration in range(num_iterations):
            iteration = self.start_iteration + local_iteration
            optimizer.zero_grad(set_to_none=True)

            if local_iteration % steps_per_epoch == 0:
                perm = torch.randperm(n_scan, device=scan_device)

            batch_slot = local_iteration % steps_per_epoch
            start = batch_slot * batch_size
            end = min(start + batch_size, n_scan)
            batch_idx = perm[start:end]

            neel = NeelObject(self.C, self.A1, self.A2)
            J = neel.build_jones(self.theta, self.phi)
            obj = JonesObject(J)

            probes = []
            for angle in self.pol_angles:
                rad = np.deg2rad(angle)
                jones_vec = torch.tensor([np.cos(rad) + 0j, np.sin(rad) + 0j], dtype=cdtype, device=self.device)
                probes.append(Probe(self.probe_amplitude, jones_vec, fluence=self.fluence, normalized=True))

            model = ForwardModel(obj, Propagator(), Detector())
            I_pred = model.simulate_all(probes, self.scan, position_indices=batch_idx)
            I_meas_batch = self.I_meas.index_select(1, batch_idx.to(self.I_meas.device))


            Lx = torch.sin(self.theta) * torch.cos(self.phi)
            Ly = torch.sin(self.theta) * torch.sin(self.phi)
            Lz = torch.cos(self.theta)

            # 2. Compute spatial gradients using finite differences
            # Differences along the X-axis (width)
            dx_Lx = torch.diff(Lx, dim=-1)
            dx_Ly = torch.diff(Ly, dim=-1)
            dx_Lz = torch.diff(Lz, dim=-1)

            # Differences along the Y-axis (height)
            dy_Lx = torch.diff(Lx, dim=-2)
            dy_Ly = torch.diff(Ly, dim=-2)
            dy_Lz = torch.diff(Lz, dim=-2)

            grad_mag_x = torch.sqrt(dx_Lx**2 + dx_Ly**2 + dx_Lz**2 + 1e-8) # 1e-8 prevents NaN in sqrt at 0
            grad_mag_y = torch.sqrt(dy_Lx**2 + dy_Ly**2 + dy_Lz**2 + 1e-8)

            loss_x = torch.sum(grad_mag_x**2)
            loss_y = torch.sum(grad_mag_y**2)

            gradient_loss = loss_x + loss_y
            loss = torch.mean((torch.sqrt(I_pred + eps) - torch.sqrt(I_meas_batch + eps)) ** 2)+ 1e-5*gradient_loss
            #loss = torch.mean(I_pred - I_meas_batch * torch.log(I_pred + eps))#+1e-5*gradient_loss
            
            loss.backward()
            optimizer.step()

            self.loss_history.append(loss.item())
            print(
                f"Iter {iteration:4d} | Loss = {loss.item():.6e} | "
                f"Batch positions: {int(batch_idx[0])}-{int(batch_idx[-1])} (size={batch_idx.numel()})"
            )

            if iteration % 100 == 0:
                plot_probe_maps(self.probe_amplitude.detach().cpu().numpy(), self.Lx, self.Ly)
                plot_theta_phi_maps(
                    self.theta.detach().cpu().numpy(),
                    self.phi.detach().cpu().numpy(),
                    self.Lx,
                    self.Ly,
                    theta_cmap='magma',
                    phi_cmap=self.RGB_scale,
                    label_axes=True,
                )

            if checkpoint_path is not None and checkpoint_every > 0 and ((iteration + 1) % checkpoint_every == 0):
                self.start_iteration = iteration + 1
                self.save(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path} at iteration {iteration + 1}")

        self.start_iteration = self.start_iteration + num_iterations

        if checkpoint_path is not None:
            self.save(checkpoint_path)
            print(f"Final checkpoint saved to {checkpoint_path}")

        return {
            "theta": self.theta.detach(),
            "phi": self.phi.detach(),
            "probe_amplitude": self.probe_amplitude.detach(),
            "C": self.C.detach(),
            "A1": self.A1.detach(),
            "A2": self.A2.detach(),
            "loss_history": list(self.loss_history),
        }



