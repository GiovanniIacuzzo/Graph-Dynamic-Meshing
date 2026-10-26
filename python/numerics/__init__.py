"""
numerics/__init__.py
Moduli numerici Python di riferimento per Graph-Dynamic-Meshing (GDM).
"""

from numerics.pde import AdvectionDiffusion2DMMS, ManufacturedConvectionDiffusion2D
from numerics.fem import (
    ConvectionDiffusionFEM2D,
    compute_nodal_gradient,
    compute_field_gradients,
    _get_triangle_quadrature_degree_5,
)
from numerics.ale import ALESolver2D
from numerics.remap import remap_p1_solution
from numerics.remap_l2 import remap_l2_projection
from numerics.dynamic_solver import DynamicPDESimulator, compute_continuous_errors

__all__ = [
    "AdvectionDiffusion2DMMS",
    "ManufacturedConvectionDiffusion2D",
    "ConvectionDiffusionFEM2D",
    "compute_nodal_gradient",
    "compute_field_gradients",
    "_get_triangle_quadrature_degree_5",
    "ALESolver2D",
    "remap_p1_solution",
    "remap_l2_projection",
    "DynamicPDESimulator",
    "compute_continuous_errors",
]