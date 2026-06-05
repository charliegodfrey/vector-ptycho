# vector-ptycho — Physics-Informed Vector Ptychography (PIVP)

Standard vector ptychography solves for a general Jones matrix using polarization diversity and polarization analysis after the object.

Here, we instead embed the physics of XMLD scattering directly into the forward model, reducing the number of unknowns and enabling reconstruction without polarization analysis.

![PIVP soft X-ray experimental setup](images/PIVP%20soft%20xray%20experimental%20setup.png)

## Key Ideas

- The sample (object) is represented by pixels, each described by a complex Jones matrix.
- This Jones matrix acts differently on different polarization states.
- The Jones matrix is constructed from a Néel field parameterized by $\theta$ and $\phi$.
- Several probes are required, each with a different Jones vector (polarization).
- A propagator transports the exit wavefield from the object plane to the detector.
- A tensor of diffraction patterns is collected with shape `(N_probes, N_positions, H, W)`.
- Autodifferentiation (via PyTorch) is used to optimize the object and probe by minimizing the difference between measured and simulated diffraction patterns.

## Physics-Informed Constraints

The physics-informed aspect of this algorithm comes from several sources:

- The unknowns are reduced by constraining the magnetic scattering tensor (Jones matrix) using the Néel vector polar angles.
- The loss function penalizes sharp gradients in the Néel field, reflecting exchange stiffness in the magnetic material.
- The loss may also penalize energetically unfavorable Néel vector orientations arising from magnetocrystalline anisotropy.

## Forward Model

The complex Jones matrix of the object acts on a Jones vector (probe) to produce an exit wave immediately after the object.

The object $\mathbf{O}$ is translated by $\boldsymbol{\delta}_j$ and $k$ indexes the polarization state and $j$ indexes the scan position.

The exit wave is then

```math
\mathbf{\Psi}_{jk}(\mathbf{x}_1)
=
\mathbf{O}(\mathbf{x}_1 - \boldsymbol{\delta}_j)
\mathbf{p}_{jk}(\mathbf{x}_1)
=
\begin{pmatrix}
\rho_{xx}(\mathbf{x}_1 - \boldsymbol{\delta}_j)
&
\rho_{xy}(\mathbf{x}_1 - \boldsymbol{\delta}_j)
\\
\rho_{yx}(\mathbf{x}_1 - \boldsymbol{\delta}_j)
&
\rho_{yy}(\mathbf{x}_1 - \boldsymbol{\delta}_j)
\end{pmatrix}
\begin{pmatrix}
p_{jk;x}(\mathbf{x}_1)
\\
p_{jk;y}(\mathbf{x}_1)
\end{pmatrix}
```

## XMLD-Based Jones Matrix

Since the goal is to reconstruct the Néel field
$\phi(\mathbf{x}_1)$ and $\theta(\mathbf{x}_1)$,
the Jones matrix can be written directly in terms of the Néel vector direction.

- Diagonal elements contain contributions from charge scattering and XMLD.
- Off-diagonal elements describe polarization mixing due to XMLD scattering.

For spherical symmetry a magnetic ion with magnetisation unit vector $\mathbf{\hat{m}}$ has the scattering factor $F$ for an incident polarisation $\epsilon_{in}$ and a scattered polarisation $\epsilon_{out}$ (see https://doi.org/10.1103/PhysRevB.82.094403):
```math
F_{\epsilon_{\mathrm{in}}\epsilon_{\mathrm{out}}}=F^{(0)}\left(\boldsymbol{\epsilon}_{\mathrm{in}}\cdot\boldsymbol{\epsilon}_{\mathrm{out}}^{*}\right)+F^{(1)}\left(\boldsymbol{\epsilon}_{\mathrm{in}}\times\boldsymbol{\epsilon}_{\mathrm{out}}^{*}\right)\cdot\hat{\mathbf{m}}+F^{(2)}\left[\left(\boldsymbol{\epsilon}_{\mathrm{out}}^{*}\cdot\hat{\mathbf{m}}\right)\left(\boldsymbol{\epsilon}_{\mathrm{in}}\cdot\hat{\mathbf{m}}\right)-\frac{1}{3}\left(\boldsymbol{\epsilon}_{\mathrm{in}}\cdot\boldsymbol{\epsilon}_{\mathrm{out}}^{*}\right)\right]
```
where $F^{(0)}$, $F^{(1)}$ and $F^{(2)}$ are the complex scattering constants associated with Thompson scattering, XMCD and XMLD in the soft x-ray regime.

Taking the sum of two antiferromagnetically coupled ions an example Jones matrix for a Néel vector $l_x = \sin\theta\cos\phi$, $l_y = \sin\theta\sin\phi$, $l_z = \cos\theta$ is:

```math
\mathbf{O}
=
2
\begin{pmatrix}
\frac{2}{3}
-
F^{(0)}
+
F^{(2)}
(\sin\theta \cos\phi)^2
&
F^{(2)}
\sin^2\theta
\cos\phi
\sin\phi
\\
F^{(2)}
\sin^2\theta
\cos\phi
\sin\phi
&
\frac{2}{3}
-
F^{(0)}
+
F^{(2)}
(\sin\theta \sin\phi)^2
\end{pmatrix}
```

## Algorithm
![Algorithm](images/PIVP%20Algorithm.png)

## Notebooks

Jupyter notebooks and their descriptions:

1. `Reconstructing_a_simulated_Neel_field.ipynb` - Simulate a Neel field, a probe and a complete dataset. Then run a full Ptycho reconstruction of probe and Neel field.
2. `Resolution_test_structure.ipynb` - Plot an example Neel field.
3. `Cosine_similarity.ipynb` - Compare the cosine similarity between two Neel vector fields.