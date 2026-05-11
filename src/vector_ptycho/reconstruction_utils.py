import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import os

from vector_ptycho.utils import *
from vector_ptycho.plotting_utils import *
from vector_ptycho.Neel_field_sim_utils import *


class PtychoReconstructionTrainer:
    def __init__(
        self,
        scan,
        scan_ref,
        pol_angles,
        I_meas,
        H,
        W,
        initial_l,
        initial_probe_amplitude,
        C,
        A1,
        A2,
        shifts,
        device=None,
        loss_prefactors=None,
        optimizer_params=None,
        Lx=None,
        Ly=None,
        R_probe=None,
        R=None,
        fluence=None,
        phi_cmap='twilight',
        theta_cmap='magma',
        object_randomisation=False,
        probe_randomisation=False,
        requires_grad_flags=None,
        alternate_optimization=False
    ):
        """
        Initialize the ptychographic reconstruction trainer with full support for 
        object and probe optimization.
        
        Parameters
        ----------
        scan : ScanTrajectory
            Scan positions.
        pol_angles : list of float
            List of polarisation angles in degrees for each probe.
        I_meas : torch.Tensor
            Measured diffraction intensities, shape (N_probes, N_scan, Hdet, Wdet)
        H, W : int
            Object map size.
        initial_l : torch.Tensor
            Initial guess for the neel vector in Cartesian coordinates, shape (3, H, W).
        initial_probe_amplitude : torch.Tensor
            Initial guess for the probe amplitude, shape (H, W).
        C, A1, A2 : torch.Tensor or complex
            XMLD/Jones constants.
        shifts : torch.Tensor
            Shifts for each probe and position, shape (N_probes, N_scan, 2).
        device : torch.device or str
            Device to use.
        loss_prefactors : dict
            Prefactors for different loss components.
        optimizer_params : dict
            Learning rates for different parameter groups.
        Lx, Ly : float
            Object dimensions.
        R_probe : float
            Probe radius for regularization.
        R : torch.Tensor
            Radial distance grid.
        fluence : float or torch.Tensor
            Target fluence.
        plot_update : callable
            Function to call for live plot updates.
        phi_cmap : colormap
            Colormap for phi visualization.
        theta_cmap : str
            Colormap name for theta visualization.
        object_randomisation : bool
            Whether to apply randomisation to object.
        probe_randomisation : bool
            Whether to apply randomisation to probe.
        alternate_optimization : bool
            Whether to use alternate optimization strategy.
        """
        self.scan = scan
        self.scan_ref = scan_ref
        self.pol_angles = pol_angles
        self.I_meas = I_meas
        self.H = H
        self.W = W
        self.l = initial_l
        self.probe_amplitude = initial_probe_amplitude
        self.C = C
        self.A1 = A1
        self.A2 = A2
        self.shifts = shifts
        self.Lx = Lx
        self.Ly = Ly
        self.R_probe = R_probe
        self.R = R
        self.fluence = fluence
        self.phi_cmap = phi_cmap
        self.theta_cmap = theta_cmap
        self.alternate_optimization = alternate_optimization

        self.device = device or I_meas.device

        self.loss_prefactors = loss_prefactors or {
            'sqrt_amp_pf': 0.0, 
            'gradient_pf': 0.0,
            'anisotropy_pf': 0.0, 
            'probe_localisation_pf': 0.0,
            'STXM_pf': 1.0,
            'probe_size_pf': 0.0
        }
        self.optimizer_params = optimizer_params or {
            'l_lr': 1e-1, 
            'CA_lr': 1e-5,
            'probe_lr': 1e-1,
            'shifts_lr': 1e-1
        }


        self.requires_grad_flags = requires_grad_flags or{
            'l': True,
            'C': True,
            'A1': True,
            'A2': True,
            'shifts': True,
            'probe_amplitude': True
        }

        self.eps = 1e-6
        self.cdtype = torch.complex64
        self.jones_vectors = self._build_all_probe_jones_vectors()
        self.sqrt_meas_sum = torch.sqrt(torch.sum(self.I_meas, dim=(-2, -1)) + self.eps) #Used in STXM loss calculation.
        # Parameters
        self.initialize_learnable_parameters()

        self._init_optimizers()

        self.loss_history = []
        self.start_iter = 0

    def initialize_learnable_parameters(self):
        """Initialize learnable parameters."""
        self.l = torch.nn.Parameter(self.l.to(self.device))
        self.probe_amplitude = torch.nn.Parameter(self.probe_amplitude.to(self.device))
        self.shifts = torch.nn.Parameter(self.shifts.to(self.device).float())

        if isinstance(self.C, torch.Tensor):
            self.C = torch.nn.Parameter(self.C.to(self.device))
        else:
            self.C = torch.nn.Parameter(torch.tensor(self.C, dtype=self.cdtype, device=self.device))
        
        if isinstance(self.A1, torch.Tensor):
            self.A1 = torch.nn.Parameter(self.A1.to(self.device))
        else:
            self.A1 = torch.nn.Parameter(torch.tensor(self.A1, dtype=self.cdtype, device=self.device))
        
        if isinstance(self.A2, torch.Tensor):
            self.A2 = torch.nn.Parameter(self.A2.to(self.device))
        else:
            self.A2 = torch.nn.Parameter(torch.tensor(self.A2, dtype=self.cdtype, device=self.device))

    def _init_optimizers(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW([
            {"params": [self.l], "lr": self.optimizer_params['l_lr']},
            {"params": [self.C, self.A1, self.A2], "lr": self.optimizer_params['CA_lr']},
            {"params": [self.probe_amplitude], "lr": self.optimizer_params['probe_lr']},
            {"params": [self.shifts], "lr": self.optimizer_params['shifts_lr']}
        ])

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5,
            patience=500,
            threshold=1e-3,
            min_lr=1e-6,
        )

    def save_checkpoint(self, path, iteration):
        """Save checkpoint with all model state and optimizer state."""
        checkpoint = {
            "iteration": iteration,
            "l": self.l.detach().cpu(),
            "probe_amplitude": self.probe_amplitude.detach().cpu(),
            "scan_positions": self.scan.positions.detach().cpu(),  # Assuming scan is serializable. If not, consider saving its parameters instead.
            "shifts": self.scan.shifts.detach().cpu() if hasattr(self.scan, 'shifts') else None,
            "C": self.C.detach().cpu(),
            "A1": self.A1.detach().cpu(),
            "A2": self.A2.detach().cpu(),
            "shifts": self.shifts.detach().cpu(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "optimizer_params": self.optimizer_params,
            "loss_prefactors": self.loss_prefactors,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None            
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint):
        """Load checkpoint and restore all state."""
        self.l = checkpoint["l"].to(self.device)
        self.probe_amplitude = checkpoint["probe_amplitude"].to(self.device)
        self.C = checkpoint["C"].to(self.device)
        self.A1 = checkpoint["A1"].to(self.device)
        self.A2 = checkpoint["A2"].to(self.device)
        self.shifts = checkpoint["shifts"].to(self.device)
        self.scan.positions = checkpoint["scan_positions"].to(self.device)
        if hasattr(self.scan, 'shifts') and checkpoint["scan_shifts"] is not None:
            self.scan.shifts = checkpoint["scan_shifts"].to(self.device)
        #self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        torch.set_rng_state(checkpoint["rng_state"])
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

        self.loss_history = checkpoint.get("loss_history", [])
        self.start_iter = checkpoint["iteration"] + 1

    def _build_probe_jones_vector(self, angle):
        """Build Jones vector for a given polarisation angle.
        -------
        Parameters
        angle : float
            Polarisation angle in degrees.
         Returns
         ------
         torch.Tensor             Jones vector of shape (2,) for the given polarisation angle."""
        rad = np.deg2rad(angle)
        return torch.tensor(
            [np.cos(rad) + 0j, np.sin(rad) + 0j], 
            dtype=self.cdtype, 
            device=self.device
            )
    
    def _build_all_probe_jones_vectors(self):
        """Build Jones vectors for all polarisation angles."""
        jones_vectors = []
        for angle in self.pol_angles:
            jones_vec = self._build_probe_jones_vector(angle)
            jones_vectors.append(jones_vec)
        return jones_vectors
    
    def _build_normalized_components(self):
        """Build normalized Cartesian components from l."""
        l_mag = torch.linalg.norm(self.l, dim=0).clamp_min(1e-8)
        lx_norm, ly_norm, lz_norm = self.l / l_mag
        return lx_norm, ly_norm, lz_norm, l_mag

    def _build_probes(self, normalized=True):
        """
        Build probe array for all polarisation angles.
        Uses the probe Jones vectors aleady pre-built.
        """
        probes = []
        for i, angle in enumerate(self.pol_angles):
            jones_vec = self.jones_vectors[i]
            probes.append(
                Probe(
                    self.probe_amplitude,
                    jones_vec,
                    fluence=self.fluence,
                    normalized=normalized,
                )
            )
        return probes

    def compute_loss(self, I_pred):
        """
        Compute total loss with multiple components.
        
        Parameters
        ----------
        I_pred : torch.Tensor
            Predicted diffraction intensities.
        iteration : int
            Current iteration (used for scheduling loss prefactors).
            
        Returns
        -------
        torch.Tensor
            Total loss value.
        """
        loss = 0.0
        I_pred_safe = torch.clamp(I_pred, min=0.0)

        # Amplitude-based ptychographic loss
        if self.loss_prefactors['sqrt_amp_pf'] > 0:
            loss = self.loss_prefactors['sqrt_amp_pf'] * torch.mean(
                (torch.sqrt(I_pred_safe + self.eps) - torch.sqrt(self.I_meas + self.eps)) ** 2
            )

        if self.loss_prefactors['STXM_pf'] > 0:
            pred_sum = torch.sum(I_pred_safe, dim=(-2, -1))
            meas_sum = torch.sum(self.I_meas, dim=(-2, -1))
            
            loss = loss + self.loss_prefactors['STXM_pf'] * torch.sum(
                (torch.sqrt(pred_sum + self.eps) - self.sqrt_meas_sum) ** 2
            )
        if np.abs(self.loss_prefactors['anisotropy_pf']) > 0.0:
            loss = loss + self.loss_prefactors['anisotropy_pf'] * torch.mean(self.l[2]**2) # Positive prefactor will favour IP alignment.
        
        if np.abs(self.loss_prefactors['gradient_pf']) > 0.0:
            dy, dx = torch.gradient(self.l, dim=(-2, -1))
            gradient_loss = torch.sum(torch.sqrt(dx**2 + dy**2 + self.eps))
            loss = loss + self.loss_prefactors['gradient_pf'] * gradient_loss

        if np.abs(self.loss_prefactors['probe_size_pf']) > 0.0:
            probe_size_loss = torch.sum(torch.abs(self.probe_amplitude) * (self.R>self.R_probe))
            loss = loss + self.loss_prefactors['probe_size_pf'] * probe_size_loss

        if np.abs(self.loss_prefactors['probe_localisation_pf']) > 0.0:
            probe_localisation_loss = torch.sum(torch.abs(self.probe_amplitude) * (self.R > self.R_probe)*self.R**2) # This loss term penalises amplitude at large R, which helps to keep the probe localised and prevent it from drifting during optimization.
            loss = loss + self.loss_prefactors['probe_localisation_pf'] * probe_localisation_loss
        return loss

    def train(self, num_iterations, checkpoint_out_path=None, save_every=50, plot_filename=None):
        """
        Full training loop with complex scheduling and optional randomization.
        
        Parameters
        ----------
        num_iterations : int
            Number of iterations to train.
        checkpoint_out_path : str or None
            Path to save checkpoints.
        save_every : int
            Save checkpoint every N iterations.
        plot_filename : str or None
            Filename for saving plots during training.
        """

        plot_update = create_live_plotter(self.Lx, self.Ly, phi_cmap=self.phi_cmap, theta_cmap=self.theta_cmap)

        alternate_optimization_counter = 0
        active_param_names = [
            param_name for param_name, requires_grad in self.requires_grad_flags.items() if requires_grad
        ]
        # count the number of parameters that require gradients
        num_grad_params = len(active_param_names)

        if self.start_iter == 0:
            print('Normalising the probe amplitude to match the target fluence before starting training.')
            with torch.no_grad():
                scale = torch.sqrt(self.fluence / torch.sum(torch.abs(self.probe_amplitude) ** 2))
                self.probe_amplitude.mul_(scale)

        for iteration in range(self.start_iter, self.start_iter + num_iterations):

            if self.alternate_optimization and num_grad_params > 0:
                active_param_name = active_param_names[alternate_optimization_counter % num_grad_params]
                for param_name in self.requires_grad_flags:
                    param = getattr(self, param_name)
                    param.requires_grad_(param_name == active_param_name)
                alternate_optimization_counter += 1
            else:
                for param_name, requires_grad in self.requires_grad_flags.items():
                    param = getattr(self, param_name)
                    param.requires_grad_(requires_grad)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass
            lx_norm, ly_norm, lz_norm, _ = self._build_normalized_components()

            # Build Jones object from current parameters
            neel = NeelObject(self.C, self.A1, self.A2)
            J = neel.build_jones_from_cartesian(lx=lx_norm, ly=ly_norm, lz=lz_norm)
            obj = JonesObject(J)

            # Build probes
            probes = self._build_probes(normalized=True)
            self.scan = ScanTrajectory(self.scan.positions_unshifted, shifts=self.shifts) # Update scan with current shifts
            # Forward model
            model = ForwardModel(obj, Propagator(), Detector())
            I_pred = model.simulate_all(probes, self.scan)
            loss = self.compute_loss(I_pred)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(loss)

            self.loss_history.append(loss.item())

            # Print learning rates periodically
            if iteration % 20 == 0:
                print([group['lr'] for group in self.optimizer.param_groups])
            '''
            # Apply randomization early in training to escape local minima
            if iteration % 8 == 0 and iteration < 30:
                with torch.no_grad():
                    if self.probe_randomisation:
                        self.probe_amplitude.data += (
                            torch.rand((self.H, self.W), device=self.device) * 
                            torch.max(torch.abs(self.probe_amplitude)) * 0.25
                        )
                    if self.object_randomisation:
                        for i in range(3):
                            noise = torch.randn_like(self.l[i])
                            self.l[i].data += 1.5 * noise
            '''
            # Save checkpoint
            '''
            if checkpoint_out_path and iteration % save_every == 0:
                self.save_checkpoint(checkpoint_out_path, iteration)
            '''
            # Update plots
            if iteration % 20 == 0:
                theta = torch.acos(torch.clamp(lz_norm, -1.0 + 1e-6, 1.0 - 1e-6))
                phi = torch.atan2(ly_norm, lx_norm)
                
                # Compute fluence from probe
                fluence_calc = torch.sum(torch.abs(self.probe_amplitude) ** 2)
                print(
                    f"Iter {iteration:4d} | Loss = {loss.item():.6e} | "
                    f"C, A1, A2: {self.C.item():.6e}, {self.A1.item():.6e}, {self.A2.item():.6e} | "
                    f"Fluence: {fluence_calc.item():.6e} | "
                    f"Active param: {active_param_names[(alternate_optimization_counter - 1) % num_grad_params] if self.alternate_optimization else 'All'}"
                )
                
                plot_update(self.probe_amplitude, theta, phi, loss, self.scan, self.scan_ref, save_filename=plot_filename)
        if checkpoint_out_path:
            self.save_checkpoint(checkpoint_out_path, iteration)

    def get_results(self):
        """Get reconstructed parameters and loss history."""
        return {
            "l": self.l.detach(),
            "probe_amplitude": self.probe_amplitude.detach(),
            "C": self.C.detach(),
            "A1": self.A1.detach(),
            "A2": self.A2.detach(),
            "shifts": self.shifts.detach(),
            "loss_history": self.loss_history,
            "optimizer_params": self.optimizer_params,
            "loss_prefactors": self.loss_prefactors,
        }
