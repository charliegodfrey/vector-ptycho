# vector-ptycho - Physics-Informed Magnetic Ptychography (PIMP)
Standard vector ptychography solves for a general Jones matrix using polarization diversity and analysis after the object.

Here, we instead embed the physics of XMLD scattering directly into the forward model, reducing the number of unknowns and enabling reconstruction without polarization analysis.

* The sample (object) is now represented by pixels each of which is a complex Jones Matrix. This Jones matrix is able to act on different polarisations differently
* The Jones Matrix is constructed from a Neel field (parametrised by $\theta$ and $\phi$) - the goal.
* Several probes are now needed, each with a different Jones vector (polarisation).
* A propagator is defined that propagates the exit wave-field just after the object to the detector.
* A tensor of diffraction patterns are now 'collected' of shape (N_probes, N_positions, H, W) where H and W are the size of the detector.
* Autodifferentiation (using PyTorch) is used to find the object and probe that minimises the loss (difference between the measured diffraction patterns and the diffraction patterns that would be produced by the current probes and object).

The Physics-Informed part of this algorithm comes from several parts:
* We are reducing the unknowns in the problem by assuming the form of the Magnetic Scattering Tensor (the Jones Matrix) in terms of the Neel vector polar angles.
* Also the loss penalises sharp gradients in the Neel field - this is constrained by the exchange stiffness of the magnetic material under inspection.
* The loss can also penalise Neel vector orientations that may be energetically unfavourable - magnetocrystalline magnetic anisotropy.

The complex Jones matrix (of the object) acts on a Jones vector (probe) to produce an exit wave just after the object. We have a probe which gets translated $P_{jk}(\textbf{x}_1) = P_{k}(\textbf{x}_1 -\delta_j)$ which corresponds to the $k^{th}$ polarisation state at the $j^{th}$ position.
$$
\mathbf{\Psi_{jk}(\mathbf{x}_1)} = \textbf{O}(\mathbf{x}_1-\boldsymbol{\delta}_j)\textbf{p}_{jk}(\mathbf{x}_1) =
\begin{pmatrix}
    \rho_{xx}(\mathbf{x}_1-\boldsymbol{\delta}_j) & \rho_{xy}(\mathbf{\mathbf{x}_1-\boldsymbol{\delta}_j}) \\
    \rho_{yx}(\mathbf{x}_1-\boldsymbol{\delta}_j) & \rho_{yy}(\mathbf{x}_1-\boldsymbol{\delta}_j)
\end{pmatrix}
\begin{pmatrix}
    p_{jk;x}(\mathbf{x}_1) \\
    p_{jk;y}(\mathbf{x}_1)
\end{pmatrix}
$$

Now since we are interested in constructing the Neel field on the sample $\phi(\textbf{x}_1)$ and $\theta(\textbf{x}_1)$ we can write the Jones matrix in terms of the Neel vector direction. On-diagonal elements are a combination of charge scattering $C$ and XMLD. Offdiagonal elements are XMLD scattering of polarisation into different channels. An example of such a matrix could be the following for a Neel vector in spherical symmetry:
$$
\textbf{O} = 2 \cdot
\begin{pmatrix}
    \frac{2}{3} - F^{(0)} + F^{(2)}(\sin\theta\cos \phi)^2 & F^{(2)}\sin^2\theta\cos\phi\sin\phi \\
    F^{(2)}\sin^2\theta\cos\phi\sin\phi&\frac{2}{3} - F^{(0)} + F^{(2)}(\sin\theta\sin \phi)^2
\end{pmatrix}
$$
