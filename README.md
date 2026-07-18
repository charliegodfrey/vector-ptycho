# vector-ptycho — Physics-Informed Vector Ptychography (PIVP)
## Charlie Godfrey
Standard vector ptychography solves for a general Jones matrix using polarisation diversity **and** polarisation analysis after the object. In the soft X-ray regime the downstream analyser is the problem: soft X-ray polarisers are multilayer Bragg reflectors that deliver high polarisation purity only within a narrow angular window — fundamentally incompatible with the broad diffraction cone recorded in a ptychographic measurement — and their reflectance falls from ~50% at 100 eV to ~1% at 700 eV, imposing a severe dose penalty on an already dose-limited technique.

**PIVP** embeds the physics of resonant XMLD scattering directly into the forward model. The Jones matrix is parameterised by the local Néel vector rather than being left unconstrained, which reduces the number of unknowns and removes the need for any polarisation analysis after the sample.

![PIVP soft X-ray experimental setup](images/PIVP%20soft%20xray%20experimental%20setup.png)

## Key Ideas

- The object is represented pixel-wise by a complex $2\times2$ Jones matrix, which acts differently on different polarisation states.
- The Jones matrix is constructed from a Néel field parameterised by two scalar fields, $\theta(\mathbf{x}_1)$ and $\phi(\mathbf{x}_1)$.
- Several probes are required, each with a different Jones vector (polarisation angle $\chi_k$).
- A propagator transports the exit wavefield from the object plane to the detector.
- A tensor of diffraction patterns is collected with shape `(N_probes, N_positions, H, W)`.
- Autodifferentiation (PyTorch) optimises the object and probe by minimising the difference between measured and simulated diffraction patterns.

## Notation

| Symbol | Meaning |
|---|---|
| $\mathbf{x}_1$ | Object-plane coordinate |
| $\mathbf{x}_2$, $\mathbf{f} = \mathbf{x}_2/\lambda z$ | Detector coordinate, spatial frequency |
| $j$, $k$ | Scan-position index, probe-polarisation index |
| $\boldsymbol{\delta}_j$ | Scan position (object translation) |
| $\mathbf{O}$ | Object Jones matrix ($2\times2$, complex) |
| $\mathbf{p}_k$ | Probe Jones vector |
| $\boldsymbol{\Psi}_{jk}$, $\hat{\boldsymbol{\Psi}}_{jk}$ | Exit wave, propagated (detector-plane) wave |
| $\mathbf{H}_l$ | Analyser Jones matrix — **not required in PIVP** |
| $F^{(0)},F^{(1)},F^{(2)}$ | Complex scattering factors: charge, XMCD, XMLD |
| $\hat{\mathbf{m}}$, $\hat{\mathbf{l}}$ | Magnetic moment direction, Néel unit vector |
| $\theta,\phi$ | Polar angles of the Néel vector |
| $\chi_k$ | Probe linear-polarisation angle |
| $\lambda$, $z$ | Wavelength, sample–detector distance |

## Forward Model

### 1. Object–probe interaction

The complex Jones matrix of the object acts on a Jones vector (probe) to produce the exit wave immediately after the object. The object $\mathbf{O}$ is translated by $\boldsymbol{\delta}_j$:

```math
\boldsymbol{\Psi}_{jk}(\mathbf{x}_1)
=
\mathbf{O}(\mathbf{x}_1 - \boldsymbol{\delta}_j)\,
\mathbf{p}_{k}(\mathbf{x}_1)
=
\begin{pmatrix}
J_{xx} & J_{xy} \\
J_{yx} & J_{yy}
\end{pmatrix}
\begin{pmatrix}
p_{k;x} \\
p_{k;y}
\end{pmatrix}
```

where the argument $(\mathbf{x}_1-\boldsymbol{\delta}_j)$ of the Jones-matrix elements is suppressed on the right-hand side for brevity.

### 2. Propagation to the detector

The two orthogonal polarisation components are propagated separately to the detector plane by a diagonal matrix of Fourier transforms:

```math
\hat{\boldsymbol{\Psi}}_{jk}(\mathbf{f})
=
\begin{pmatrix}
\mathcal{F}_{\mathbf{x}_1\rightarrow\mathbf{f}} & 0 \\
0 & \mathcal{F}_{\mathbf{x}_1\rightarrow\mathbf{f}}
\end{pmatrix}
\boldsymbol{\Psi}_{jk}(\mathbf{x}_1)
```

### 3. Detection — no analyser

In **conventional** vector ptychography the propagated field passes through an analyser $\mathbf{H}_l$ before the detector records $I_{jkl} = \lVert \mathbf{H}_l \hat{\boldsymbol{\Psi}}_{jk} \rVert^2$. It is precisely this analyser that is experimentally problematic in the soft X-ray regime.

In **PIVP** the object is constrained by physics, so the analyser can be dispensed with entirely and the detector simply records the incoherent sum of the two propagated polarisation channels:

```math
I_{jk}(\mathbf{f})
=
\left\lVert \hat{\boldsymbol{\Psi}}_{jk}(\mathbf{f}) \right\rVert^2
=
\left| \hat{\Psi}_{jk;x}(\mathbf{f}) \right|^2
+
\left| \hat{\Psi}_{jk;y}(\mathbf{f}) \right|^2
```

## XMLD-Based Jones Matrix

### Scattering tensor

The anisotropic optical response near a resonance is encoded in the complex scattering tensor $F$. For a magnetic ion of normalised moment direction $\hat{\mathbf{m}}$ in **spherical symmetry**, electric-dipole scattering gives three terms ([Haverkort et al., PRB **82**, 094403](https://doi.org/10.1103/PhysRevB.82.094403)):

```math
\begin{aligned}
F ={}& F^{(0)}\left(\boldsymbol{\epsilon}_{\mathrm{in}}\cdot\boldsymbol{\epsilon}_{\mathrm{out}}^{*}\right)
+F^{(1)}\left(\boldsymbol{\epsilon}_{\mathrm{in}}\times\boldsymbol{\epsilon}_{\mathrm{out}}^{*}\right)\cdot\hat{\mathbf{m}} \\
&+F^{(2)}\left[\left(\boldsymbol{\epsilon}_{\mathrm{out}}^{*}\cdot\hat{\mathbf{m}}\right)\left(\boldsymbol{\epsilon}_{\mathrm{in}}\cdot\hat{\mathbf{m}}\right)-\tfrac{1}{3}\left(\boldsymbol{\epsilon}_{\mathrm{in}}\cdot\boldsymbol{\epsilon}_{\mathrm{out}}^{*}\right)\right]
\end{aligned}
```

The complex, energy-dependent prefactors encompass all spectroscopic detail (crystal field, exchange, spin–orbit coupling), while the tensor structure is fixed by symmetry:

- $F^{(0)}$ — charge (Thomson/anomalous) scattering
- $F^{(1)}$ — **odd** in $\hat{\mathbf{m}}$ → XMCD
- $F^{(2)}$ — **even** in $\hat{\mathbf{m}}$ → XMLD

Because the XMLD term enters as $\hat{\mathbf{m}}\otimes\hat{\mathbf{m}}$, it is invariant under $\hat{\mathbf{m}}\rightarrow-\hat{\mathbf{m}}$ — exactly as required for imaging the Néel vector of an antiferromagnet.

### Small-angle limit: collapsing to a $2\times2$ Jones matrix

In a general scattering experiment the polarisation vectors are decomposed into $\boldsymbol{\sigma}$ and $\boldsymbol{\pi}$ components, defined **separately** for the incident and scattered X-rays:

```math
\begin{gathered}
\boldsymbol{\pi}_{\mathrm{in}} \parallel (\mathbf{k}_{\mathrm{in}} \times \boldsymbol{\sigma}) \\
\boldsymbol{\pi}_{\mathrm{out}} \parallel (\mathbf{k}_{\mathrm{out}} \times \boldsymbol{\sigma}) \\
\boldsymbol{\sigma} \parallel (\mathbf{k}_{\mathrm{in}} \times \mathbf{k}_{\mathrm{out}})
\end{gathered}
```

In the **small-angle scattering limit** $\mathbf{k}_{\mathrm{in}} \approx \mathbf{k}_{\mathrm{out}}$, so the $\sigma$–$\pi$ coordinate systems for the incident and scattered X-rays coincide. Setting $\boldsymbol{\sigma}=\hat{\mathbf{x}}$ and $\boldsymbol{\pi}_{\mathrm{in}}=\boldsymbol{\pi}_{\mathrm{out}}=\hat{\mathbf{y}}$, the scattering factor collapses to a single $2\times2$ matrix acting on the two-component transverse polarisation state — i.e. the object Jones matrix that enters the forward model above.

### Single sublattice

Evaluating the decomposition explicitly in this geometry, for a moment $\hat{\mathbf{m}} = (\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta)$:

```math
\mathbf{O}_{\mathbf{m}_1}
=
\begin{pmatrix}
F^{(0)} + F^{(2)}\left(\sin^2\theta\cos^2\phi - \tfrac{1}{3}\right)
&
-F^{(1)}\cos\theta + F^{(2)}\sin^2\theta\cos\phi\sin\phi
\\
F^{(1)}\cos\theta + F^{(2)}\sin^2\theta\cos\phi\sin\phi
&
F^{(0)} + F^{(2)}\left(\sin^2\theta\sin^2\phi - \tfrac{1}{3}\right)
\end{pmatrix}
```

### Antiferromagnetic average

The second sublattice of a collinear AFM is obtained by the substitution $\theta \rightarrow \pi - \theta$, $\phi \rightarrow \phi + \pi$. This leaves the $F^{(0)}$ and $F^{(2)}$ terms unchanged but **reverses the sign of the odd $F^{(1)}$ (XMCD) contributions**. Averaging the two antiparallel sublattices therefore cancels $F^{(1)}$ and yields the physics-informed object:

```math
\mathbf{O}
=
\begin{pmatrix}
F^{(0)} + F^{(2)}\left(\sin^2\theta\cos^2\phi - \tfrac{1}{3}\right)
&
F^{(2)}\sin^2\theta\cos\phi\sin\phi
\\
F^{(2)}\sin^2\theta\cos\phi\sin\phi
&
F^{(0)} + F^{(2)}\left(\sin^2\theta\sin^2\phi - \tfrac{1}{3}\right)
\end{pmatrix}
```

where $\theta$ and $\phi$ are now the polar angles of the **Néel vector**. The entire object is parameterised by just two spatially varying scalar fields, together with the complex scattering factors $F^{(0)}$ and $F^{(2)}$ (either assumed known or treated as free parameters in the model).

- **Diagonal** elements contain contributions from charge scattering and XMLD.
- **Off-diagonal** elements describe polarisation mixing due to XMLD scattering — these are *not* negligible, which is the core motivation for PIVP.

## Physics-Informed Constraints

The physics-informed aspect of this algorithm comes from several sources:

- The unknowns are reduced by constraining the magnetic scattering tensor (Jones matrix) using the Néel vector polar angles $\theta,\phi$ (2 real fields, instead of 8 real fields for an unconstrained complex $2\times2$ matrix).
- The Néel vector is constrained to unit length at every pixel after each iteration.
- *(Optional)* The loss function may penalise sharp gradients in the Néel field, reflecting exchange stiffness in the magnetic material.
- *(Optional)* The loss may penalise energetically unfavourable Néel orientations arising from magnetocrystalline anisotropy.

## Reconstruction

### Loss function

At high spatial frequency the scattered intensity is weak, so detected counts are dominated by Poisson shot noise and a least-squares loss weights the data inappropriately. The natural choice is the Poisson negative log-likelihood; in practice we minimise its widely used square-root approximation:

```math
\mathcal{L}_S = \sum_j\sum_{m}\left(\sqrt{h_{m,j}} - \sqrt{y_{m,j}}\right)^{2}
```

where $y_{m,j}$ is the measured photon count at detector pixel $m$ for position $j$, and $h_{m,j}$ is the corresponding modelled intensity.

### Optimisation schedule

Reconstruction proceeds in four stages:

1. **500 iterations** minimising the MSE between summed experimental and summed simulated diffraction patterns (a "STXM-like" loss) — gives a low-resolution starting estimate limited by the probe size, but improves later convergence.
2. **300 iterations** of probe-only optimisation under the square-root loss.
3. **5000 iterations** of simultaneous object and probe optimisation.
4. **10 × 200 iterations** of "warm" restarts at high learning rate, encouraging the optimiser to escape local minima (reminiscent of simulated annealing).

A probe-localisation penalty proportional to $R^2$ outside a chosen radius suppresses the translational degeneracy common to ptychography and keeps the reconstructed probe compact. In later stages of reconstruction this can be removed.

> **Note:** diffraction patterns are normalised by their peak intensity, with the probe amplitude scaled consistently. Without this normalisation the reconstruction can diverge, with the object growing more opaque as the probe photon dose increases to compensate.

**Runtime:** ~45 min for a full reconstruction (121 scan positions, 4 probe polarisations, 512x512 detector pixels) on a 24-core 3.7 GHz CPU / 64 GB RAM / RTX 2000 Ada GPU.

## Evaluation Metrics

### Cosine similarity

Reconstruction fidelity against a known ground truth:

```math
\Delta = \frac{1}{N}\sum_{i=1}^{N}\left|\,\hat{\mathbf{l}}_i^{\mathrm{recon}} \cdot \hat{\mathbf{l}}_i^{\mathrm{true}}\,\right|
```

The modulus accounts for the $\mathbf{l}\rightarrow-\mathbf{l}$ equivalence of the Néel vector, so a perfect reconstruction gives $\Delta=1$ while a random field gives $\Delta=0.5$.

### Vector Fourier ring correlation (vFRC)

A naive component-wise FRC ignores the coupling between Néel components imposed by $|\mathbf{l}|=1$ and spuriously registers the $\mathbf{l}\rightarrow-\mathbf{l}$ ambiguity as high-frequency signal. Both problems are resolved by working with the traceless, $\mathbf{l}\rightarrow-\mathbf{l}$-invariant $Q$-tensor from liquid-crystal theory, $Q_{ij} = l_i l_j - \tfrac{1}{3}\delta_{ij}$:

```math
\text{vFRC}(q) = \frac{\left|\sum_{\mathbf{q}\in\mathrm{ring}} \tilde{\mathbf{Q}}_1(\mathbf{q}) : \tilde{\mathbf{Q}}_2^{*}(\mathbf{q})\right|}{\sqrt{\sum_{\mathbf{q}\in\mathrm{ring}} |\tilde{\mathbf{Q}}_1(\mathbf{q})|^2 \; \sum_{\mathbf{q}\in\mathrm{ring}} |\tilde{\mathbf{Q}}_2(\mathbf{q})|^2}}
```

where $\tilde{\mathbf{Q}}_1 : \tilde{\mathbf{Q}}_2^{*} = \sum_{ij}\tilde{Q}_{1,ij}\tilde{Q}_{2,ij}^{*}$ is the Frobenius inner product. The two fields are reconstructed from datasets differing only in their Poisson noise seed. Resolution is taken as the frequency at which the vFRC falls below the half-bit threshold.

### Edge fitting

Sharp edges in the reconstructed field are fitted to $y = A\tanh\!\left(\frac{x-x_0}{w}\right) + B$; the FWHM of the derivative of this step, $1.763\,w$, defines the resolution.

## Default Simulation Parameters

Chosen to approximate the Diamond I08-1 beamline and the planned Diamond-II CSXID beamline.

| Parameter | Value |
|---|---|
| Wavelength $\lambda$ | 1.74 nm |
| Sample–detector distance $z$ | 70 mm |
| Detector pixel size | 26.0 × 26.0 µm² |
| Detector pixels | 512 × 512 |
| FZP diameter | 200 µm |
| FZP inner radius (central stop) | 80 µm |
| Outermost zone width | 30 nm |
| Defocus distance | 20 µm |
| Object pixel size | 9.15 nm |
| Object size | 4.68 × 4.68 µm² |
| $F^{(0)}$ | $1.0 + 1.0i$ |
| $F^{(2)}$ | $0.1 + 0.1i$ |
| Scan positions | 121 |
| Polarisations $\chi_k$ | 0°, 30°, 60°, 90° |
| Probe diameter | 1.0 µm |
| Probe overlap | 50% |
| Probe photons per exposure | $10^{11}$ – $10^{13}$ |

The factor-of-ten ratio $|F^{(2)}|/|F^{(0)}| = 0.1$ is realistic for resonant magnetic scattering, where charge scattering dominates the magnetic contribution (e.g. Fe₂O₃ at the Fe $L$ edges).

## Algorithm

![Algorithm](images/PIVP%20Algorithm.png)

### Classes overview

Main classes in `src/vector_ptycho`:

- **`PtychoReconstructionTrainer`** — Orchestrates the reconstruction and optimisation loop. Manages learnable parameters (object `l`, probe amplitude, `F_scat`, shifts), optimisers/schedulers, loss terms, checkpointing, and training utilities.
- **`NeelObject`** — Physics-driven parameterisation of the object. Stores scattering factors `F_scat` and builds a complex Jones matrix from Néel vector angles or Cartesian components via `build_jones` / `build_jones_from_cartesian`.
- **`JonesObject`** — Holds a full (unconstrained) Jones matrix `J` and applies it to a `JonesField` via `apply(field)`.
- **`Probe`** — Encapsulates probe amplitude and Jones vector; supports normalisation and shifting. Provides `field()` and `shifted(dy, dx)`.
- **`Propagator`** — Far-field Fourier propagation for a `JonesField` via the Fourier operators (`F` / `iF`), exposing `propagate(field)`.
- **`Detector`** — Computes measured intensity from a `JonesField` (`intensity(field)`) and can optionally add Poisson noise (`add_poisson_noise`).

## Notebooks

1. **`Reconstructing_a_simulated_Neel_field.ipynb`** — Simulate a Néel field, a probe and a complete dataset, then run a full ptychographic reconstruction of probe and Néel field.
2. **`Resolution_test_structure.ipynb`** — Plot an example Néel field.
3. **`Cosine_similarity.ipynb`** — Compare the cosine similarity between two Néel vector fields.
4. More notebooks are contained in **`Successfull fluence run 17_07_2026 RndSeed 5`** where a systematic sequence of experiments with different photon doses per diffraction pattern were performed.

## Installation

Ensure you have [UV](https://docs.astral.sh/uv/) installed. Then:

```bash
git clone https://github.com/charliegodfrey/vector-ptycho.git
cd vector-ptycho
uv sync
```

This creates a virtual environment and installs all dependencies. You can then open and run the Jupyter notebooks in the `notebooks` folder.


## Citation

If you use this code, please cite:

```bibtex
@article{godfrey_pivp,
  title   = {Physics-informed vector ptychography for imaging vector order parameters},
  author  = {Godfrey, Charles and Radaelli, Paolo G.},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## References

- Haverkort, Hollmann, Krug & Tanaka, *Symmetry analysis of magneto-optical effects*, [Phys. Rev. B **82**, 094403 (2010)](https://doi.org/10.1103/PhysRevB.82.094403) — scattering tensor decomposition.
- Schäfers *et al.*, *Soft-x-ray polarimeter with multilayer optics*, [Appl. Opt. **38**, 4074 (1999)](https://doi.org/10.1364/AO.38.004074) — soft X-ray polariser performance.

## License

<!-- TODO: add a LICENSE file and state it here (MIT / BSD-3 / GPL-3). -->
