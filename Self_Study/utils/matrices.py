"""Utility functions for displaying matrices and tensors as LaTeX in Jupyter notebooks."""

import numpy as np
from IPython.display import display, Math, Latex


def _array_to_latex(arr):
    """Convert a numpy array to a LaTeX matrix string."""
    arr = np.atleast_1d(arr)

    if arr.ndim == 0:
        return str(arr.item())
    elif arr.ndim == 1:
        # Display as a column vector
        rows = [str(x) for x in arr]
        body = " \\\\\n".join(rows)
        return r"\begin{bmatrix}" + "\n" + body + "\n" + r"\end{bmatrix}"
    elif arr.ndim == 2:
        rows = []
        for row in arr:
            rows.append(" & ".join(str(x) for x in row))
        body = " \\\\\n".join(rows)
        return r"\begin{bmatrix}" + "\n" + body + "\n" + r"\end{bmatrix}"
    else:
        # For higher-order tensors, show each 2D slice
        slices = []
        for idx in np.ndindex(arr.shape[:-2]):
            label = f"[:, :, {', '.join(str(i) for i in idx)}]" if idx else ""
            slice_latex = _array_to_latex(arr[idx])
            if label:
                slices.append(f"\\text{{{label}}}\\quad {slice_latex}")
            else:
                slices.append(slice_latex)
        return "\\quad ".join(slices)


def show_boxed_tensor_latex(tensor, name=None):
    """Display a numpy array as a boxed LaTeX-rendered matrix in a Jupyter notebook.

    Parameters
    ----------
    tensor : array_like
        The tensor or matrix to display. Will be converted to a numpy array.
    name : str, optional
        An optional label to show before the matrix (e.g. ``"A"`` renders as ``A = [...]``).
    """
    tensor = np.asarray(tensor)
    latex_str = _array_to_latex(tensor)

    if name is not None:
        latex_str = f"{name} = {latex_str}"

    display(Math(r"\boxed{" + latex_str + r"}"))
