import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


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
        amp = torch.roll(self.amplitude, (dy, dx), dims=(0,1))
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
        Jxy = self.A2 * (sin_p * cos_p * sin_t**2)
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


