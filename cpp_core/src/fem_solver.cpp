#include "gdm/fem_solver.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/matrix_tools.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <map>
#include <stdexcept>

namespace gdm {

ConvectionDiffusionFEM2D::ConvectionDiffusionFEM2D(
    const MeshState& mesh,
    const ManufacturedConvectionDiffusion2D& pde,
    const LinearSolverParameters& solver_params,
    const StabilizationParameters& stabilization_params)
    : mesh_(&mesh),
      pde_(pde),
      solver_params_(solver_params),
      stabilization_params_(stabilization_params),
      quadrature_(create_dunavant_degree5_quadrature()) {
    solver_params_.validate();
    stabilization_params_.validate();
    assemble_static_operators();
}

void ConvectionDiffusionFEM2D::rebuild_for_mesh(const MeshState& mesh) {
    mesh_ = &mesh;
    assemble_static_operators();
}

void ConvectionDiffusionFEM2D::assemble_static_operators() {
    const auto& dof_handler = mesh_->dof_handler();
    const std::size_t n_dofs = dof_handler.n_dofs();

    dealii::DynamicSparsityPattern dsp(n_dofs, n_dofs);
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern_.copy_from(dsp);

    mass_matrix_.reinit(sparsity_pattern_);
    stiffness_matrix_.reinit(sparsity_pattern_);
    convection_matrix_.reinit(sparsity_pattern_);

    dealii::FEValues<2> fe_values(
        mesh_->mapping(),
        mesh_->fe(),
        quadrature_,
        dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = mesh_->fe().n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_.size();

    dealii::FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> cell_stiff(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> cell_conv(dofs_per_cell, dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    const auto vel = pde_.velocity();
    const dealii::Tensor<1, 2> b_vec({vel[0], vel[1]});
    const double nu = pde_.nu();

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_mass = 0.0;
        cell_stiff = 0.0;
        cell_conv = 0.0;

        for (unsigned int q = 0; q < n_q_points; ++q) {
            const double jxw = fe_values.JxW(q);
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const double phi_i = fe_values.shape_value(i, q);
                const auto grad_i = fe_values.shape_grad(i, q);
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const double phi_j = fe_values.shape_value(j, q);
                    const auto grad_j = fe_values.shape_grad(j, q);

                    cell_mass(i, j) += phi_i * phi_j * jxw;
                    cell_conv(i, j) += phi_i * (b_vec * grad_j) * jxw;
                    cell_stiff(i, j) += nu * (grad_i * grad_j) * jxw;
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                mass_matrix_.add(local_dof_indices[i], local_dof_indices[j], cell_mass(i, j));
                convection_matrix_.add(local_dof_indices[i], local_dof_indices[j], cell_conv(i, j));
                stiffness_matrix_.add(local_dof_indices[i], local_dof_indices[j], cell_stiff(i, j));
            }
        }
    }

    cached_geometry_revision_ = mesh_->geometry_revision();
}

std::vector<double> ConvectionDiffusionFEM2D::assemble_forcing_vector(double t) const {
    const auto& dof_handler = mesh_->dof_handler();
    const std::size_t n_dofs = dof_handler.n_dofs();
    std::vector<double> f_py(n_dofs, 0.0);

    dealii::FEValues<2> fe_values(
        mesh_->mapping(),
        mesh_->fe(),
        quadrature_,
        dealii::update_values | dealii::update_quadrature_points | dealii::update_JxW_values);

    const unsigned int dofs_per_cell = mesh_->fe().n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_.size();
    std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q = 0; q < n_q_points; ++q) {
            const auto& pt = fe_values.quadrature_point(q);
            const double f_val = pde_.evaluate_forcing(pt[0], pt[1], t);
            const double jxw = fe_values.JxW(q);
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const std::size_t py_i = mesh_->dof_to_py_node(local_dof_indices[i]);
                f_py[py_i] += f_val * fe_values.shape_value(i, q) * jxw;
            }
        }
    }
    return f_py;
}

std::vector<double> ConvectionDiffusionFEM2D::step_backward_euler(
    const std::vector<double>& u_n,
    double dt,
    double t_next,
    double* out_linear_residual,
    double* out_assembly_seconds,
    double* out_solve_seconds,
    double* out_rhs_norm,
    double* out_solver_tolerance) const {
    if (u_n.size() != mesh_->num_nodes()) {
        throw std::invalid_argument("step_backward_euler: dimensione u_n non valida.");
    }
    if (dt <= 0.0 || !std::isfinite(dt)) {
        throw std::invalid_argument("step_backward_euler: dt deve essere > 0.");
    }
    if (cached_geometry_revision_ != mesh_->geometry_revision()) {
        throw std::logic_error("step_backward_euler: geometria mesh modificata senza rebuild_for_mesh.");
    }

    const auto assembly_start = std::chrono::steady_clock::now();
    const std::size_t n_dofs = mesh_->num_nodes();
    dealii::SparseMatrix<double> system_matrix(sparsity_pattern_);
    system_matrix.copy_from(mass_matrix_);
    system_matrix *= (1.0 / dt);
    system_matrix.add(1.0, convection_matrix_);
    system_matrix.add(1.0, stiffness_matrix_);

    dealii::Vector<double> u_n_deal(n_dofs);
    for (std::size_t py_i = 0; py_i < n_dofs; ++py_i) {
        u_n_deal[mesh_->py_node_to_dof(py_i)] = u_n[py_i];
    }

    dealii::Vector<double> system_rhs(n_dofs);
    mass_matrix_.vmult(system_rhs, u_n_deal);
    system_rhs *= (1.0 / dt);

    const std::vector<double> f_py = assemble_forcing_vector(t_next);
    for (std::size_t py_i = 0; py_i < n_dofs; ++py_i) {
        system_rhs[mesh_->py_node_to_dof(py_i)] += f_py[py_i];
    }

    // Residual-based SUPG for the fully discrete backward-Euler P1 operator.
    // On affine P1 triangles, Delta(u_h)=0 elementwise; the strong residual
    // therefore contains the transient, advective and forcing terms.
    if (stabilization_params_.use_supg) {
        dealii::FEValues<2> fe_values(
            mesh_->mapping(), mesh_->fe(), quadrature_,
            dealii::update_values | dealii::update_gradients |
            dealii::update_quadrature_points | dealii::update_JxW_values);
        const unsigned int dofs_per_cell = mesh_->fe().n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_.size();
        dealii::FullMatrix<double> cell_supg(dofs_per_cell, dofs_per_cell);
        dealii::Vector<double> cell_rhs(dofs_per_cell);
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
        const auto vel = pde_.velocity();
        const std::array<double, 2> beta = {vel[0], vel[1]};

        for (const auto& cell : mesh_->dof_handler().active_cell_iterators()) {
            fe_values.reinit(cell);
            cell_supg = 0.0;
            cell_rhs = 0.0;
            cell->get_dof_indices(local_dof_indices);

            std::array<std::array<double, 2>, 3> grad_phi{};
            std::array<std::array<double, 2>, 3> vertices{};
            for (unsigned int i = 0; i < 3; ++i) {
                const auto grad = fe_values.shape_grad(i, 0);
                grad_phi[i] = {grad[0], grad[1]};
                const auto vertex = cell->vertex(i);
                vertices[i] = {vertex[0], vertex[1]};
            }
            const double speed = std::hypot(beta[0], beta[1]);
            const double h_beta = directional_element_length(grad_phi, beta, vertices);
            const double tau = supg_tau(dt, speed, h_beta, pde_.nu(), stabilization_params_);

            for (unsigned int q = 0; q < n_q_points; ++q) {
                double u_old_q = 0.0;
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    u_old_q += u_n_deal[local_dof_indices[j]] * fe_values.shape_value(j, q);
                }
                const auto& point = fe_values.quadrature_point(q);
                const double forcing = pde_.evaluate_forcing(point[0], point[1], t_next);
                const double jxw = fe_values.JxW(q);

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const auto grad_i = fe_values.shape_grad(i, q);
                    const double streamline_test = beta[0] * grad_i[0] + beta[1] * grad_i[1];
                    cell_rhs[i] += tau * (u_old_q / dt + forcing) * streamline_test * jxw;
                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const auto grad_j = fe_values.shape_grad(j, q);
                        const double streamline_trial = beta[0] * grad_j[0] + beta[1] * grad_j[1];
                        const double transient_trial = fe_values.shape_value(j, q) / dt;
                        cell_supg(i, j) += tau * (transient_trial + streamline_trial) *
                                           streamline_test * jxw;
                    }
                }
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                system_rhs[local_dof_indices[i]] += cell_rhs[i];
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_supg(i, j));
                }
            }
        }
    }

    // Imposizione forte Dirichlet: eliminazione di righe/colonne e correzione del termine noto
    std::map<dealii::types::global_dof_index, double> boundary_values;
    const auto& nodes = mesh_->nodes();
    for (std::int64_t b_py : mesh_->boundary_nodes()) {
        const auto idx = static_cast<std::size_t>(b_py);
        const auto dof = mesh_->py_node_to_dof(idx);
        boundary_values[dof] = pde_.evaluate_solution(nodes[idx][0], nodes[idx][1], t_next);
    }

    dealii::Vector<double> solution_deal(n_dofs);
    dealii::MatrixTools::apply_boundary_values(
        boundary_values, system_matrix, solution_deal, system_rhs, true);

    const double rhs_norm = system_rhs.l2_norm();
    const double solver_tolerance = std::max(solver_params_.abs_tol,
                                             solver_params_.rel_tol * rhs_norm);
    if (out_rhs_norm != nullptr) {
        *out_rhs_norm = rhs_norm;
    }
    if (out_solver_tolerance != nullptr) {
        *out_solver_tolerance = solver_tolerance;
    }

    if (out_assembly_seconds != nullptr) {
        *out_assembly_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - assembly_start).count();
    }
    const auto solve_start = std::chrono::steady_clock::now();
    if (solver_params_.use_direct_solver) {
        dealii::SparseDirectUMFPACK direct_solver;
        direct_solver.initialize(system_matrix);
        direct_solver.vmult(solution_deal, system_rhs);
    } else {
        dealii::SolverControl solver_control(
            static_cast<unsigned int>(solver_params_.max_iterations), solver_tolerance);
        // With right preconditioning, SolverControl monitors the unpreconditioned
        // ||b-Ax||_2, matching the independently reported residual. The
        // default remains left preconditioning to preserve legacy runs.
        dealii::SolverGMRES<dealii::Vector<double>>::AdditionalData gmres_data(
            static_cast<unsigned int>(solver_params_.gmres_krylov_dim),
            solver_params_.right_preconditioning, true);
        dealii::SolverGMRES<dealii::Vector<double>> solver(solver_control, gmres_data);

        dealii::PreconditionSSOR<dealii::SparseMatrix<double>> preconditioner;
        preconditioner.initialize(system_matrix, solver_params_.ssor_omega);
        solver.solve(system_matrix, solution_deal, system_rhs, preconditioner);
    }

    if (out_solve_seconds != nullptr) {
        *out_solve_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - solve_start).count();
    }
    if (out_linear_residual != nullptr) {
        dealii::Vector<double> res(n_dofs);
        system_matrix.vmult(res, solution_deal);
        res -= system_rhs;
        *out_linear_residual = res.l2_norm();
    }

    std::vector<double> u_next(n_dofs);
    for (std::size_t py_i = 0; py_i < n_dofs; ++py_i) {
        u_next[py_i] = solution_deal[mesh_->py_node_to_dof(py_i)];
    }
    return u_next;
}

std::vector<double> ConvectionDiffusionFEM2D::multiply_mass_matrix(
    const std::vector<double>& u) const {
    const std::size_t n = mesh_->num_nodes();
    if (u.size() != n) {
        throw std::invalid_argument("multiply_mass_matrix: dimensione non valida.");
    }
    dealii::Vector<double> u_deal(n), mu_deal(n);
    for (std::size_t i = 0; i < n; ++i) {
        u_deal[mesh_->py_node_to_dof(i)] = u[i];
    }
    mass_matrix_.vmult(mu_deal, u_deal);
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = mu_deal[mesh_->py_node_to_dof(i)];
    }
    return out;
}

double ConvectionDiffusionFEM2D::compute_discrete_mass(const std::vector<double>& u) const {
    const auto mu = multiply_mass_matrix(u);
    double sum = 0.0;
    for (double val : mu) {
        sum += val;
    }
    return sum;
}

std::vector<std::array<double, 2>> compute_field_gradients(
    const MeshState& mesh,
    const std::vector<double>& u,
    const ManufacturedConvectionDiffusion2D* pde,
    double t,
    bool exact_gradient) {
    const auto& nodes = mesh.nodes();
    const auto& cells = mesh.cells();
    const std::size_t n_nodes = nodes.size();

    if (u.size() != n_nodes) {
        throw std::invalid_argument("compute_field_gradients: dimensione di u non coerente con i nodi.");
    }

    std::vector<std::array<double, 2>> grad(n_nodes, {0.0, 0.0});
    if (exact_gradient) {
        if (pde == nullptr) {
            throw std::invalid_argument("compute_field_gradients: pde richiesto per exact_gradient=true.");
        }
        for (std::size_t i = 0; i < n_nodes; ++i) {
            grad[i] = pde->evaluate_gradient(nodes[i][0], nodes[i][1], t);
        }
        return grad;
    }

    // Ricostruzione nodale pesata per l'area degli elementi adiacenti (come in numerics/fem.py)
    std::vector<double> nodal_area(n_nodes, 0.0);
    for (const auto& c : cells) {
        const auto i1 = static_cast<std::size_t>(c[0]);
        const auto i2 = static_cast<std::size_t>(c[1]);
        const auto i3 = static_cast<std::size_t>(c[2]);

        const double x1 = nodes[i1][0], y1 = nodes[i1][1];
        const double x2 = nodes[i2][0], y2 = nodes[i2][1];
        const double x3 = nodes[i3][0], y3 = nodes[i3][1];

        const double det_j = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
        const double area = 0.5 * std::abs(det_j);
        const double inv_det = 1.0 / det_j;

        const double dphi1_dx = (y2 - y3) * inv_det;
        const double dphi2_dx = (y3 - y1) * inv_det;
        const double dphi3_dx = (y1 - y2) * inv_det;

        const double dphi1_dy = (x3 - x2) * inv_det;
        const double dphi2_dy = (x1 - x3) * inv_det;
        const double dphi3_dy = (x2 - x1) * inv_det;

        const double gx_e = u[i1] * dphi1_dx + u[i2] * dphi2_dx + u[i3] * dphi3_dx;
        const double gy_e = u[i1] * dphi1_dy + u[i2] * dphi2_dy + u[i3] * dphi3_dy;

        for (std::size_t idx : {i1, i2, i3}) {
            grad[idx][0] += area * gx_e;
            grad[idx][1] += area * gy_e;
            nodal_area[idx] += area;
        }
    }

    for (std::size_t i = 0; i < n_nodes; ++i) {
        const double denom = std::max(nodal_area[i], 1.0e-14);
        grad[i][0] /= denom;
        grad[i][1] /= denom;
    }
    return grad;
}

} // namespace gdm