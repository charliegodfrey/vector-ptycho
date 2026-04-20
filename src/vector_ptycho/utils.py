import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_vector_color_map():
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
    plt.imshow(gradient, aspect='auto', cmap=RGB_scale)
    return RGB_scale

def _to_numpy(x):
    """Safely convert torch / array-like to numpy."""
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)

def plot_theta_phi_maps(theta, phi, Lx, Ly,
                       positions=None,
                       theta_cmap='magma',
                       phi_cmap=RGB_scale,
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
        torch.fft.fft2(torch.fft.fftshift(x, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1)
    )

def iF(x):
    return torch.fft.ifftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'),
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
    def __init__(self, amplitude, jones_vector):
        self.amplitude = amplitude
        self.jones_vector = jones_vector

    def field(self):
        Ex = self.amplitude * self.jones_vector[0]
        Ey = self.amplitude * self.jones_vector[1]
        return JonesField(Ex, Ey)

    def shifted(self, dy, dx):
        amp = torch.roll(self.amplitude, shifts=(dy, dx), dims=(0, 1))
        return Probe(amp, self.jones_vector)


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
    def intensity(self, field: JonesField):
        return torch.abs(field.Ex)**2 + torch.abs(field.Ey)**2


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

    def simulate_all(self, probes, scan):
        data = []

        for probe in probes:
            probe_data = []

            for dy, dx in scan.positions.tolist():
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

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.show()

    if return_torch:
        theta = torch.tensor(theta_np, dtype=torch.float32, device=out_device)
        phi = torch.tensor(phi_np, dtype=torch.float32, device=out_device)
        return theta, phi, Mx, My, Mz

    return theta_np, phi_np, Mx, My, Mz