"""
oracle/field.py
Definizione dei campi scalari analitici di riferimento per Graph-Dynamic Meshing (GDM).

Implementa la Gaussian bump bidimensionale e le sue derivate esatte:
    u(x, y) = A * exp( -((x - mu_x)^2 + (y - mu_y)^2) / (2 * sigma^2) )
    grad_u(x, y) = [du/dx, du/dy]^T
    laplacian(x, y) = d^2u/dx^2 + d^2u/dy^2

Modulo completamente analitico, disaccoppiato da GNN e solutori numerici.
"""

from typing import Dict, Any, Tuple
import numpy as np


class GaussianBumpField:
    """
    Rappresenta un campo scalare gaussiano isotropo 2D valutabile analiticamente.
    """

    def __init__(
        self,
        mu_x: float = 0.5,
        mu_y: float = 0.5,
        sigma: float = 0.05,
        amplitude: float = 1.0
    ):
        """
        Inizializza i parametri del campo gaussiano.

        Input:
            mu_x: coordinata x del centro della campana
            mu_y: coordinata y del centro della campana
            sigma: deviazione standard / larghezza caratteristica (sigma > 0)
            amplitude: ampiezza massima A al vertice (x = mu_x, y = mu_y)
        """
        assert sigma > 0.0, f"sigma deve essere strettamente positivo, ottenuto: {sigma}"
        self.mu_x = float(mu_x)
        self.mu_y = float(mu_y)
        self.sigma = float(sigma)
        self.amplitude = float(amplitude)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Restituisce i parametri del campo per la tracciabilità nei dataset e nei manifest."""
        return {
            "field_type": "gaussian_bump_2d",
            "mu_x": self.mu_x,
            "mu_y": self.mu_y,
            "sigma": self.sigma,
            "amplitude": self.amplitude,
        }

    def _validate_coordinates(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Garantisce che x e y siano array NumPy 1D contigui con le stesse dimensioni."""
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)

        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError(
                f"Le coordinate devono essere vettori 1D. Ottenuto: x.shape={x_arr.shape}, y.shape={y_arr.shape}"
            )
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"x e y devono avere la stessa lunghezza. Ottenuto: {x_arr.shape[0]} vs {y_arr.shape[0]}"
            )

        return x_arr, y_arr

    def evaluate(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Valuta congiuntamente il valore del campo, il gradiente e il laplaciano su un insieme di punti.

        Input:
            x: coordinate cartesiane x dei punti, shape [Nq]
            y: coordinate cartesiane y dei punti, shape [Nq]

        Output:
            u: valori scalari del campo analitico, shape [Nq], dtype float64
            grad_u: componenti cartesiane del gradiente [du/dx, du/dy], shape [Nq, 2], dtype float64
            laplacian: valore dello scalare laplaciano Delta u, shape [Nq], dtype float64
            metadata: dizionario con i parametri fisici della gaussiana
        """
        x_arr, y_arr = self._validate_coordinates(x, y)

        # Distanze relative rispetto al centro
        dx = x_arr - self.mu_x
        dy = y_arr - self.mu_y

        # Raggio al quadrato r^2 = (x - mu_x)^2 + (y - mu_y)^2
        r_sq = dx**2 + dy**2
        sigma_sq = self.sigma**2

        # 1. Campo scalare: u(x, y) = A * exp( -r^2 / (2 * sigma^2) )
        u = self.amplitude * np.exp(-0.5 * r_sq / sigma_sq)

        # 2. Gradiente analitico:
        # du/dx = - (dx / sigma^2) * u
        # du/dy = - (dy / sigma^2) * u
        inv_sigma_sq = 1.0 / sigma_sq
        grad_u_x = -dx * inv_sigma_sq * u
        grad_u_y = -dy * inv_sigma_sq * u
        grad_u = np.stack([grad_u_x, grad_u_y], axis=1)  # [Nq, 2]

        # 3. Laplaciano analitico:
        # d^2u/dx^2 = (dx^2 / sigma^4 - 1 / sigma^2) * u
        # d^2u/dy^2 = (dy^2 / sigma^4 - 1 / sigma^2) * u
        # Delta u = d^2u/dx^2 + d^2u/dy^2 = ( (dx^2 + dy^2)/sigma^4 - 2/sigma^2 ) * u
        inv_sigma_4 = inv_sigma_sq**2
        laplacian = (r_sq * inv_sigma_4 - 2.0 * inv_sigma_sq) * u

        return u, grad_u, laplacian, self.metadata

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Interfaccia diretta per la chiamata funzionale: evaluate(x, y)."""
        return self.evaluate(x, y)