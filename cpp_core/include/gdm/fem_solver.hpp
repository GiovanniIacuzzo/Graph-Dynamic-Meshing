#pragma once

#include "gdm/mesh_state.hpp"
#include "gdm/parameters.hpp"
#include "gdm/pde_model.hpp"
#include "gdm/stabilization.hpp"

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <array>
#include <cstdint>
#include <vector>

namespace gdm {

class ConvectionDiffusionFEM2D {
public:
    ConvectionDiffusionFEM2D(const MeshState& mesh,
                             const ManufacturedConvectionDiffusion2D& pde,
                             const LinearSolverParameters& solver_params = LinearSolverParameters{},
                             const StabilizationParameters& stabilization_params = StabilizationParameters{});

    void rebuild_for_mesh(const MeshState& mesh);

    std::vector<double> step_backward_euler(const std::vector<double>& u_n,
                                            double dt,
                                            double t_next,
                                            double* out_linear_residual = nullptr,
                                            double* out_assembly_seconds = nullptr,
                                            double* out_solve_seconds = nullptr,
                                            double* out_rhs_norm = nullptr,
                                            double* out_solver_tolerance = nullptr) const;

    const dealii::SparseMatrix<double>& mass_matrix() const noexcept { return mass_matrix_; }
    const dealii::SparseMatrix<double>& stiffness_matrix() const noexcept { return stiffness_matrix_; }
    const dealii::SparseMatrix<double>& convection_matrix() const noexcept { return convection_matrix_; }
    const dealii::SparsityPattern& sparsity_pattern() const noexcept { return sparsity_pattern_; }

    std::vector<double> multiply_mass_matrix(const std::vector<double>& u) const;
    double compute_discrete_mass(const std::vector<double>& u) const;

private:
    void assemble_static_operators();
    std::vector<double> assemble_forcing_vector(double t) const;

    const MeshState* mesh_{nullptr};
    ManufacturedConvectionDiffusion2D pde_;
    LinearSolverParameters solver_params_;
    StabilizationParameters stabilization_params_;
    dealii::Quadrature<2> quadrature_;

    dealii::SparsityPattern sparsity_pattern_;
    dealii::SparseMatrix<double> mass_matrix_;
    dealii::SparseMatrix<double> stiffness_matrix_;
    dealii::SparseMatrix<double> convection_matrix_;
    std::uint64_t cached_geometry_revision_{0};
};

// Ricostruzione del gradiente nodale mediato per area (gradient_mode = "fem") o esatto ("exact")
std::vector<std::array<double, 2>> compute_field_gradients(
    const MeshState& mesh,
    const std::vector<double>& u,
    const ManufacturedConvectionDiffusion2D* pde = nullptr,
    double t = 0.0,
    bool exact_gradient = false);

} // namespace gdm