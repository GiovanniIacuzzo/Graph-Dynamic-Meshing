"""
numerics/pde.py
Definizione analitica della PDE di Convezione-Diffusione 2D con Manufactured Solution (MMS).

Equazione governante:
    du/dt + b · grad(u) - nu * Laplacian(u) = f(x, y, t)   su Omega = [0, 1]^2, t in [0, T]

Soluzione esatta target (ManufacturedConvectionDiffusion2D):
    u(x, y, t) = G(x, y, t) * S(x, y) * T_decay(t)
con:
    G(x, y, t) = exp( -((x - cx(t))^2 + (y - cy(t))^2) / (2 * sigma^2) )
    S(x, y)    = sin(pi * x) * sin(pi * y)
    T_decay(t) = exp( -beta * t )
    cx(t)      = 0.3 + 0.4 * (t / T)
    cy(t)      = 0.35 + 0.3 * (t / T)
"""

from typing import Tuple, Dict, Any
import numpy as np


class BasePDE:
    """Classe base per modelli PDE."""
    pass


class AdvectionDiffusion2DMMS(BasePDE):
    """
    Problema di Convezione-Diffusione 2D con soluzione fabbricata gaussiana:
        du/dt + b_x * du/dx + b_y * du/dy - nu * (d^2u/dx^2 + d^2u/dy^2) = f(x, y, t)
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
        c_y: float = 0.3,
    ):
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
        return self.x0 + self.c_x * t, self.y0 + self.c_y * t

    def evaluate_exact(
        self, x: np.ndarray, y: np.ndarray, t: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_arr = np.ascontiguousarray(x, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.float64)

        xc, yc = self._get_center(t)
        rx = x_arr - xc
        ry = y_arr - yc
        r_sq = rx**2 + ry**2
        sigma_sq = self.sigma**2

        u_exact = self.amplitude * np.exp(-0.5 * r_sq / sigma_sq)

        inv_sigma_sq = 1.0 / sigma_sq
        du_dx = -rx * inv_sigma_sq * u_exact
        du_dy = -ry * inv_sigma_sq * u_exact
        grad_exact = np.stack([du_dx, du_dy], axis=1)

        inv_sigma_4 = inv_sigma_sq**2
        laplacian_exact = (r_sq * inv_sigma_4 - 2.0 * inv_sigma_sq) * u_exact

        return u_exact, grad_exact, laplacian_exact

    def evaluate_forcing(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
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
        du_dt = u * (rx * self.c_x + ry * self.c_y) * inv_sigma_sq
        b_dot_grad = -u * (self.b[0] * rx + self.b[1] * ry) * inv_sigma_sq
        diff_term = self.nu * u * (r_sq * inv_sigma_4 - 2.0 * inv_sigma_sq)

        return du_dt + b_dot_grad - diff_term

    def evaluate_boundary(self, x_bnd: np.ndarray, y_bnd: np.ndarray, t: float) -> np.ndarray:
        u_bnd, _, _ = self.evaluate_exact(x_bnd, y_bnd, t)
        return u_bnd


class ManufacturedConvectionDiffusion2D:
    """
    Gestione analitica esatta della Manufactured Solution e del termine sorgente f(x, y, t)
    impiegata nelle campagne Stage B e Stage C.
    """

    def __init__(
        self,
        bx: float = 0.4,
        by: float = 0.2,
        nu: float = 1e-3,
        sigma: float = 0.06,
        beta: float = 0.5,
        T: float = 0.5,
    ):
        self.bx = float(bx)
        self.by = float(by)
        self.nu = float(nu)
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.T = float(T)

        self.inv_sigma2 = 1.0 / (self.sigma**2)
        self.inv_sigma4 = 1.0 / (self.sigma**4)

        # Velocità di traslazione del centro cx'(t) e cy'(t)
        self.dcx_dt = 0.4 / self.T
        self.dcy_dt = 0.3 / self.T

    def get_center(self, t: float) -> Tuple[float, float]:
        """Calcola la traiettoria del centro della campana al tempo t."""
        cx = 0.3 + 0.4 * (t / self.T)
        cy = 0.35 + 0.3 * (t / self.T)
        return cx, cy

    def evaluate_exact(
        self, x: np.ndarray, y: np.ndarray, t: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcola analiticamente la soluzione esatta e tutte le sue derivate necessarie per il forcing:
            u, u_x, u_y, u_xx, u_yy, u_t, f
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        cx, cy = self.get_center(t)
        rx = x - cx
        ry = y - cy
        r2 = rx**2 + ry**2

        # 1. Componenti base
        G = np.exp(-0.5 * r2 * self.inv_sigma2)
        S = np.sin(np.pi * x) * np.sin(np.pi * y)
        T_decay = np.exp(-self.beta * t)

        u = G * S * T_decay

        # 2. Derivate spaziali di S(x, y)
        Sx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        Sy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
        Sxx = -(np.pi**2) * S
        Syy = -(np.pi**2) * S

        # 3. Derivate spaziali di G(x, y, t)
        Gx = -rx * self.inv_sigma2 * G
        Gy = -ry * self.inv_sigma2 * G
        Gxx = ((rx * self.inv_sigma2) ** 2 - self.inv_sigma2) * G
        Gyy = ((ry * self.inv_sigma2) ** 2 - self.inv_sigma2) * G

        # 4. Derivate prime spaziali di u = G * S * T_decay
        ux = (Gx * S + G * Sx) * T_decay
        uy = (Gy * S + G * Sy) * T_decay

        # 5. Derivate seconde spaziali di u
        uxx = (Gxx * S + 2.0 * Gx * Sx + G * Sxx) * T_decay
        uyy = (Gyy * S + 2.0 * Gy * Sy + G * Syy) * T_decay

        # 6. Derivata temporale di G e u
        Gt = (rx * self.dcx_dt + ry * self.dcy_dt) * self.inv_sigma2 * G
        ut = (Gt - self.beta * G) * S * T_decay

        # 7. Termine sorgente analitico f(x, y, t)
        laplacian_u = uxx + uyy
        convection_u = self.bx * ux + self.by * uy
        f = ut + convection_u - self.nu * laplacian_u

        return u, ux, uy, uxx, uyy, ut, f

    def evaluate_forcing(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Restituisce direttamente il forcing f(x, y, t) sui punti indicati."""
        _, _, _, _, _, _, f = self.evaluate_exact(x, y, t)
        return f

    def evaluate_solution(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Restituisce il valore puntuale della soluzione esatta."""
        u, _, _, _, _, _, _ = self.evaluate_exact(x, y, t)
        return u