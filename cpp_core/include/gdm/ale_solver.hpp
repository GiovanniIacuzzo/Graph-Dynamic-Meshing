#pragma once

#include "gdm/mesh_state.hpp"
#include "gdm/parameters.hpp"
#include "gdm/pde_model.hpp"
#include "gdm/stabilization.hpp"

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <array>
#include <utility>
#include <vector>

namespace gdm {

struct ALEDiagnostics {
    double max_elem_gcl_residual{0.0};
    double algebraic_gcl_residual_linf{0.0};
    double dual_c_mesh_col_sum_linf{0.0};
    double discrete_mass_n{0.0};
    double discrete_mass_next{0.0};
    double linear_solver_residual{0.0};
    double linear_solver_rhs_norm{0.0};
    double linear_solver_tolerance{0.0};
    std::vector<double> element_gcl_residuals;
};

struct ALEStepResult {
    std::vector<double> u_next;
    ALEDiagnostics diagnostics;
};

class ALESolver2D {
public:
    explicit ALESolver2D(const ManufacturedConvectionDiffusion2D* pde = nullptr,
                         double nu = 1.0e-3,
                         double bx = 0.4,
                         double by = 0.2,
                         const LinearSolverParameters& solver_params = LinearSolverParameters{},
                         const StabilizationParameters& stabilization_params = StabilizationParameters{});

    static std::vector<MeshState::Coord2D> compute_mesh_velocity(
        const std::vector<MeshState::Coord2D>& nodes_n,
        const std::vector<MeshState::Coord2D>& nodes_next,
        double dt);

    static double compute_discrete_mass(const MeshState& mesh,
                                        const std::vector<double>& u);

    ALEStepResult step_ale(const MeshState& mesh_n,
                           const MeshState& mesh_next,
                           const std::vector<double>& u_n,
                           double dt,
                           double t_next,
                           bool pure_gcl = false,
                           bool enforce_dirichlet = true) const;

    ALEDiagnostics evaluate_gcl_diagnostics(const MeshState& mesh_n,
                                            const MeshState& mesh_next,
                                            double dt) const;

private:
    const ManufacturedConvectionDiffusion2D* pde_{nullptr};
    double nu_{1.0e-3};
    std::array<double, 2> b_{0.4, 0.2};
    LinearSolverParameters solver_params_;
    StabilizationParameters stabilization_params_;
    dealii::Quadrature<2> quadrature_;
};

} // namespace gdm