"""Adapter for evaluating the Stage-B/C manufactured solution as a static field.

`OracleObjective` consumes fields implementing evaluate(x, y) ->
(u, grad_u, laplacian, metadata).  This adapter fixes the time argument of
ManufacturedConvectionDiffusion2D and therefore guarantees that the oracle
optimizes the same spatial field used by the FEM error evaluation.
"""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np


class MMSFieldAtTime:
    """View a transient manufactured solution at one fixed time."""

    def __init__(self, pde: Any, time: float):
        self.pde = pde
        self.time = float(time)
        if not np.isfinite(self.time):
            raise ValueError("time deve essere finito.")
        self._metadata = {
            "field_type": "manufactured_convection_diffusion_2d",
            "time": self.time,
            "bx": float(self.pde.bx),
            "by": float(self.pde.by),
            "nu": float(self.pde.nu),
            "sigma": float(self.pde.sigma),
            "beta": float(self.pde.beta),
            "T": float(self.pde.T),
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def evaluate_value_gradient(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate only the exact field value and spatial gradient.

        ``OracleObjective`` uses this fast-path interface inside its repeated
        objective/gradient evaluations.  Keep it consistent with ``evaluate``
        while avoiding the unnecessary Laplacian, forcing term and metadata.
        """
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape != y_arr.shape:
            raise ValueError("x e y devono essere vettori 1D della stessa shape.")

        u, ux, uy, _uxx, _uyy, _ut, _forcing = self.pde.evaluate_exact(
            x_arr, y_arr, self.time
        )
        values = np.ascontiguousarray(u, dtype=np.float64)
        gradient = np.empty((x_arr.size, 2), dtype=np.float64)
        gradient[:, 0] = ux
        gradient[:, 1] = uy
        return values, gradient

    def evaluate(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)
        if x_arr.ndim != 1 or y_arr.ndim != 1 or x_arr.shape != y_arr.shape:
            raise ValueError("x e y devono essere vettori 1D della stessa shape.")
        u, ux, uy, uxx, uyy, _ut, _forcing = self.pde.evaluate_exact(
            x_arr, y_arr, self.time
        )
        grad = np.column_stack((ux, uy))
        laplacian = uxx + uyy
        return u, grad, laplacian, self.metadata
