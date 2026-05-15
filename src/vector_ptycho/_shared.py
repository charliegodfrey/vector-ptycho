"""Shared utilities used across multiple vector_ptycho modules."""

import torch
import numpy as np


def _to_numpy(x):
    """Safely convert torch / array-like to numpy."""
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)
