"""
numerics/pde.py
Astrazione delle equazioni differenziali alle derivate parziali (PDE)
e formulazione Manufactured Solutions (MMS) per Graph-Dynamic Meshing (GDM).

Fornisce:
    - BasePDE: classe astratta per problemi del tipo L(u) = f
    - AdvectionDiffusion2DMMS: problema 2D transiente di convezione-diffusione:
        du/dt + b * grad(u) - nu * laplacian(u) = f(x, y, t)
      con soluzione esatta analitica a struttura localizzata mobile:
        u_exact(x, y, t) = A * exp( -((x - x_c(t))^2 + (y - y_c(t))^2) / (2 * sigma^2) )
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class BasePDE(ABC):
    """
    Classe base astratta per una PDE del tipo L(u) = f su Omega x [0, T].
    """

    @abstractmethod
    def evaluate_exact(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcola la soluzione analitica esatta e le sue derivate spaziali.

        Input:
            x: coordinate x, shape [N]
            y: coordinate y, shape [N]
            t: istante temporale scalare

        Output:
            u_exact        : valori del campo, shape [N]
            grad_u_exact   : componenti cartesiane [du/dx, du/dy], shape [N, 2]
            laplacian_exact: scalare laplaciano Delta u, shape [N]
        """
        pass

    @abstractmethod
    def evaluate_forcing(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Calcola il termine di forzante forte f(x, y, t) tale che L(u_exact) = f.

        Input:
            x: coordinate x, shape [N]
            y: coordinate y, shape [N]
            t: istante temporale scalare

        Output:
            f: valore del forzamento, shape [N]
        """
        pass

    @abstractmethod
    def evaluate_boundary(
        self,
        x_bnd: np.ndarray,
        y_bnd: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Valuta i valori di Dirichlet al contorno d_Omega all'istante t.
        """
        pass


class AdvectionDiffusion2DMMS(BasePDE):
    """
    Problema di Convezione-Diffusione 2D con soluzione fabbricata (MMS):
        du/dt + b_x * du/dx + b_y * du/dy - nu * (d^2u/dx^2 + d^2u/dy^2) = f(x, y, t)
    
    Soluzione esatta:
        u(x, y, t) = A * exp( -((x - x_c(t))^2 + (y - y_c(t))^2) / (2 * sigma^2) )
    con traiettoria rettilinea:
        x_c(t) = x0 + c_x * t
        y_c(t) = y0 + c_y * t
    """

    def __init__(
        self,
        b: Tuple[float, float] = (1.0, 0.5),
        nu: float = 1e-3,
        sigma: float = 0.04,
        amplitude: float = 1.0,
        x0: float = 0.2,
        y0: float = 0.2,
        c_x: float = 0.6,
        c_y: float = 0.3
    ):
        """
        Input:
            b        : vettore di velocità convettiva (b_x, b_y)
            nu       : diffusività cinematica (nu > 0)
            sigma    : raggio / varianza del pacchetto gaussiano
            amplitude: ampiezza massima A della soluzione
            x0, y0   : centro iniziale al tempo t = 0
            c_x, c_y : componenti della velocità di traslazione del pacchetto (dot_x_c, dot_y_c)
        """
        assert nu > 0.0, f"La diffusività nu deve essere positiva, ottenuta: {nu}"
        assert sigma > 0.0, f"sigma deve essere positivo, ottenuto: {sigma}"

        self.b = np.array(b, dtype=np.float64)
        self.nu = float(nu)
        self.sigma = float(sigma)
        self.amplitude = float(amplitude)

        self.x0 = float(x0)
        self.y0 = float(y0)
        self.c_x = float(c_x)
        self.c_y = float(c_y)

    @property
    def parameters(self) -> Dict[str, Any]:
        """Parametri caratteristici della PDE per tracking e benchmark."""
        return {
            "pde_type": "advection_diffusion_2d_mms",
            "b_x": float(self.b[0]),
            "b_y": float(self.b[1]),
            "nu": self.nu,
            "sigma": self.sigma,
            "amplitude": self.amplitude,
            "x0": self.x0,
            "y0": self.y0,
            "c_x": self.c_x,
            "c_y": self.c_y,
        }

    def _get_center(self, t: float) -> Tuple[float, float]:
        """Restituisce le coordinate del centro della gaussiana al tempo t."""
        return self.x0 + self.c_x * t, self.y0 + self.c_y * t

    def evaluate_exact(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Valuta la soluzione esatta analitica, il gradiente e il laplaciano.
        """
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)

        xc, yc = self._get_center(t)
        rx = x_arr - xc
        ry = y_arr - yc
        r_sq = rx**2 + ry**2
        sigma_sq = self.sigma**2

        # 1. Campo esatto: u = A * exp( -r^2 / (2*sigma^2) )
        u_exact = self.amplitude * np.exp(-0.5 * r_sq / sigma_sq)

        # 2. Gradiente analitico: grad(u) = - (r / sigma^2) * u
        inv_sigma_sq = 1.0 / sigma_sq
        du_dx = -rx * inv_sigma_sq * u_exact
        du_dy = -ry * inv_sigma_sq * u_exact
        grad_exact = np.stack([du_dx, du_dy], axis=1)

        # 3. Laplaciano analitico: Delta u = (r^2 / sigma^4 - 2 / sigma^2) * u
        inv_sigma_4 = inv_sigma_sq**2
        laplacian_exact = (r_sq * inv_sigma_4 - 2.0 * inv_sigma_sq) * u_exact

        return u_exact, grad_exact, laplacian_exact

    def evaluate_forcing(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Calcola analiticamente la forzante forte residua:
            f = du/dt + b * grad(u) - nu * Delta u
        """
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)

        xc, yc = self._get_center(t)
        rx = x_arr - xc
        ry = y_arr - yc
        r_sq = rx**2 + ry**2

        sigma_sq = self.sigma**2
        inv_sigma_sq = 1.0 / sigma_sq
        inv_sigma_4 = inv_sigma_sq**2

        u = self.amplitude * np.exp(-0.5 * r_sq / sigma_sq)

        # Derivata temporale: du/dt = u * (rx * dot_xc + ry * dot_yc) / sigma^2
        du_dt = u * (rx * self.c_x + ry * self.c_y) * inv_sigma_sq

        # Termine avvettivo: b * grad(u) = - u * (b_x * rx + b_y * ry) / sigma^2
        b_dot_grad = -u * (self.b[0] * rx + self.b[1] * ry) * inv_sigma_sq

        # Termine diffusivo: nu * Delta u = nu * u * (r^2 / sigma^4 - 2 / sigma^2)
        diff_term = self.nu * u * (r_sq * inv_sigma_4 - 2.0 * inv_sigma_sq)

        # Bilancio forte esatto:
        forcing = du_dt + b_dot_grad - diff_term
        return forcing

    def evaluate_boundary(
        self,
        x_bnd: np.ndarray,
        y_bnd: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Valuta i valori esatti di Dirichlet non omogenei sul contorno d_Omega.
        """
        u_bnd, _, _ = self.evaluate_exact(x_bnd, y_bnd, t)
        return u_bnd