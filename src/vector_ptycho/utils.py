import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def _to_numpy(x):
    """Safely convert torch / array-like to numpy."""
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def normalise_neel(l):
    # Normalize to ensure L is always a unit vector
    mag = torch.sqrt(torch.sum(l**2, dim=0) + 1e-8)
    Lx = l[0] / mag
    Ly = l[1] / mag
    Lz = l[2] / mag
    return Lx, Ly, Lz

def cartesian_to_spherical(l):
    # Convert Cartesian to spherical coordinates (theta, phi)
    lx_norm, ly_norm, lz_norm = normalise_neel(l)
    theta = torch.acos(torch.clamp(lz_norm, -1.0 + 1e-6, 1.0 - 1e-6)) # Clamp to avoid NaNs from acos outside of [-1, 1]
    phi = torch.atan2(ly_norm, lx_norm)
    return theta, phi


def _as_device_tensor(value, *, device=device, dtype=None):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype) if (device is not None or dtype is not None) else value
    return torch.as_tensor(value, device=device, dtype=dtype)


def _shift_complex_image(image, shifts):
    """Shift a complex 2D image by sub-pixel amounts with differentiable sampling."""
    if image.ndim != 2:
        raise ValueError("image must have shape (H, W)")

    device = image.device
    height, width = image.shape

    image_ri = torch.stack([image.real, image.imag], dim=0).unsqueeze(0)
    image_ri = image_ri.expand(shifts.shape[0], -1, -1, -1)

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=image.real.dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=image.real.dtype),
        indexing='ij',
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)

    norm_x = 2.0 * shifts[:, 1] / max(width - 1, 1)
    norm_y = 2.0 * shifts[:, 0] / max(height - 1, 1)
    grid = base_grid.expand(shifts.shape[0], -1, -1, -1).clone()
    grid[..., 0] = grid[..., 0] - norm_x[:, None, None]
    grid[..., 1] = grid[..., 1] - norm_y[:, None, None]

    shifted = torch.nn.functional.grid_sample(
        image_ri,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )

    return torch.complex(shifted[:, 0], shifted[:, 1])

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
        theta = torch.acos(torch.clamp(lz, -1.0 + 1e-6, 1.0 - 1e-6))
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
    '''Defines the scan positions for ptychography.
    
    positions: tensor of shape (num_probes, num_positions, 2) containing (y, x) shifts in lab coordinates.
    shift: tensor of shape (num_probes,num_positions, 2) containing the shifts for each sampling point.
    '''
    def __init__(self, positions, shifts=None):
        self.positions_unshifted = _as_device_tensor(positions, device=device)
        if shifts is not None:
            self.shifts = _as_device_tensor(shifts, device=device)
            if not torch.is_floating_point(self.positions_unshifted) and torch.is_floating_point(self.shifts):
                self.positions_unshifted = self.positions_unshifted.to(dtype=self.shifts.dtype)
            elif torch.is_floating_point(self.positions_unshifted) and self.positions_unshifted.dtype != self.shifts.dtype:
                self.shifts = self.shifts.to(dtype=self.positions_unshifted.dtype)
            self.positions = self.positions_unshifted + self.shifts
        else:
            self.shifts = torch.zeros_like(self.positions_unshifted, device=device)
            self.positions = self.positions_unshifted


# =========================
# Forward model
# =========================
class ForwardModel:
    def __init__(self, obj, propagator, detector):
        self.obj = obj
        self.propagator = propagator
        self.detector = detector

    def simulate_all(self, probes, scan, position_indices=None):
        """
        Batched simulation over probes and scan positions.

        For each probe we precompute the shifted probe fields for every
        scan position (shape: num_positions x H x W), apply the object
        (broadcasting over positions), then perform a single batched FFT
        across all positions for GPU efficiency. Returns a tensor of
        shape (num_probes, num_positions, H, W).
        """

        if position_indices is None:
            positions = scan.positions
        else:
            idx = torch.as_tensor(position_indices, dtype=torch.long, device=scan.positions.device)
            positions = scan.positions.index_select(0, idx)

        data = []

        for i, probe in enumerate(probes):
            amp = probe.amplitude  # (H, W)

            # Build stacked shifted amplitudes for all positions with differentiable sub-pixel sampling.
            shifts = positions[i].to(dtype=amp.real.dtype)
            amps = _shift_complex_image(amp, shifts)

            # Create Jones field stacks for all positions: (P, H, W)
            j0 = probe.jones_vector[0]
            j1 = probe.jones_vector[1]
            Ex = amps * j0
            Ey = amps * j1

            # Apply object (Jones multiplication) in a batched manner.
            field = JonesField(Ex, Ey)
            field = self.obj.apply(field)

            # Batched propagation (FFT) across positions
            field = self.propagator.propagate(field)

            # Compute intensities for all positions
            I = self.detector.intensity(field)

            # I has shape (P, H, W) for this probe
            data.append(I)

        # Stack over probes -> (num_probes, num_positions, H, W)
        return torch.stack(data, dim=0)

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




