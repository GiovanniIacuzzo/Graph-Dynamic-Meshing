#include "gdm/ale_solver.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace gdm {

namespace {

constexpr std::array<std::array<double, 3>, 3> kBaseRefMass = {{
    {2.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0},
    {1.0 / 12.0, 2.0 / 12.0, 1.0 / 12.0},
    {1.0 / 12.0, 1.0 / 12.0, 2.0 / 12.0}
}};

} // namespace

ALESolver2D::ALESolver2D(const ManufacturedConvectionDiffusion2D* pde,
                         double nu,
                         double bx,
                         double by,
                         const LinearSolverParameters& solver_params)
    : pde_(pde),
      nu_(nu),
      b_{bx, by},
      solver_params_(solver_params),
      quadrature_(create_dunavant_degree5_quadrature()) {
    solver_params_.validate();
    if (pde_ != nullptr) {
        nu_ = pde_->nu();
        b_ = pde_->velocity();
    }
}

std::vector<MeshState::Coord2D> ALESolver2D::compute_mesh_velocity(
    const std::vector<MeshState::Coord2D>& nodes_n,
    const std::vector<MeshState::Coord2D>& nodes_next,
    double dt) {
    if (nodes_n.size() != nodes_next.size()) {
        throw std::invalid_argument("compute_mesh_velocity: numero di nodi incompatibile.");
    }
    if (dt <= 0.0 || !std::isfinite(dt)) {
        throw std::invalid_argument("compute_mesh_velocity: dt deve essere > 0.");
    }
    std::vector<MeshState::Coord2D> w(nodes_n.size());
    const double inv_dt = 1.0 / dt;
    for (std::size_t i = 0; i < nodes_n.size(); ++i) {
        w[i] = {(nodes_next[i][0] - nodes_n[i][0]) * inv_dt,
                (nodes_next[i][1] - nodes_n[i][1]) * inv_dt};
    }
    return w;
}

double ALESolver2D::compute_discrete_mass(const MeshState& mesh, const std::vector<double>& u) {
    const auto& nodes = mesh.nodes();
    const auto& cells = mesh.cells();
    if (u.size() != nodes.size()) {
        throw std::invalid_argument("compute_discrete_mass: dimensione u non valida.");
    }
    double mass = 0.0;
    for (const auto& c : cells) {
        const auto i1 = static_cast<std::size_t>(c[0]);
        const auto i2 = static_cast<std::size_t>(c[1]);
        const auto i3 = static_cast<std::size_t>(c[2]);
        const double det_j = (nodes[i2][0] - nodes[i1][0]) * (nodes[i3][1] - nodes[i1][1]) -
                             (nodes[i3][0] - nodes[i1][0]) * (nodes[i2][1] - nodes[i1][1]);
        const double area = 0.5 * std::abs(det_j);
        mass += (area / 3.0) * (u[i1] + u[i2] + u[i3]);
    }
    return mass;
}

ALEStepResult ALESolver2D::step_ale(const MeshState& mesh_n,
                                    const MeshState& mesh_next,
                                    const std::vector<double>& u_n,
                                    double dt,
                                    double t_next,
                                    bool pure_gcl,
                                    bool enforce_dirichlet) const {
    const std::size_t n_nodes = mesh_next.num_nodes();
    const std::size_t n_cells = mesh_next.num_cells();

    if (!mesh_n.has_same_topology(mesh_next) || u_n.size() != n_nodes) {
        throw std::invalid_argument(
            "step_ale: mesh_n e mesh_next devono avere connettivita' e boundary identici; u_n deve avere una voce per nodo.");
    }
    if (!std::isfinite(dt) || dt <= 0.0 || !std::isfinite(t_next)) {
        throw std::invalid_argument("step_ale: dt deve essere finito e > 0 e t_next deve essere finito.");
    }
    for (const double value : u_n) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("step_ale: u_n contiene valori non finiti.");
        }
    }
    mesh_n.verify_positive_jacobians();
    mesh_next.verify_positive_jacobians();

    const auto& nodes_n = mesh_n.nodes();
    const auto& nodes_next = mesh_next.nodes();
    const auto& cells = mesh_next.cells();
    const auto w_nodes = compute_mesh_velocity(nodes_n, nodes_next, dt);

    const auto& dof_handler = mesh_next.dof_handler();
    dealii::DynamicSparsityPattern dsp(n_nodes, n_nodes);
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);
    dealii::SparsityPattern sparsity;
    sparsity.copy_from(dsp);

    dealii::SparseMatrix<double> A_mat(sparsity);
    dealii::SparseMatrix<double> M_n_mat(sparsity);
    dealii::SparseMatrix<double> M_next_mat(sparsity);
    dealii::SparseMatrix<double> C_mesh_mat(sparsity);

    dealii::Vector<double> F_vec(n_nodes);

    ALEDiagnostics diag;
    diag.element_gcl_residuals.resize(n_cells, 0.0);

    const auto& q_pts = quadrature_.get_points();
    const auto& q_wts = quadrature_.get_weights();
    const std::size_t n_q = q_pts.size();

    for (std::size_t e = 0; e < n_cells; ++e) {
        const std::array<std::size_t, 3> py_idx = {
            static_cast<std::size_t>(cells[e][0]),
            static_cast<std::size_t>(cells[e][1]),
            static_cast<std::size_t>(cells[e][2])
        };
        const std::array<dealii::types::global_dof_index, 3> dofs = {
            mesh_next.py_node_to_dof(py_idx[0]),
            mesh_next.py_node_to_dof(py_idx[1]),
            mesh_next.py_node_to_dof(py_idx[2])
        };

        // 1. Geometria al punto medio X^{n+1/2}
        const double x1_m = 0.5 * (nodes_n[py_idx[0]][0] + nodes_next[py_idx[0]][0]);
        const double y1_m = 0.5 * (nodes_n[py_idx[0]][1] + nodes_next[py_idx[0]][1]);
        const double x2_m = 0.5 * (nodes_n[py_idx[1]][0] + nodes_next[py_idx[1]][0]);
        const double y2_m = 0.5 * (nodes_n[py_idx[1]][1] + nodes_next[py_idx[1]][1]);
        const double x3_m = 0.5 * (nodes_n[py_idx[2]][0] + nodes_next[py_idx[2]][0]);
        const double y3_m = 0.5 * (nodes_n[py_idx[2]][1] + nodes_next[py_idx[2]][1]);

        const double det_j_mid = (x2_m - x1_m) * (y3_m - y1_m) - (x3_m - x1_m) * (y2_m - y1_m);
        const double area_mid = 0.5 * std::abs(det_j_mid);
        const double inv_det_m = 1.0 / det_j_mid;

        const std::array<double, 3> dphi_dx_m = {
            (y2_m - y3_m) * inv_det_m, (y3_m - y1_m) * inv_det_m, (y1_m - y2_m) * inv_det_m
        };
        const std::array<double, 3> dphi_dy_m = {
            (x3_m - x2_m) * inv_det_m, (x1_m - x3_m) * inv_det_m, (x2_m - x1_m) * inv_det_m
        };

        const double div_w =
            w_nodes[py_idx[0]][0] * dphi_dx_m[0] + w_nodes[py_idx[1]][0] * dphi_dx_m[1] + w_nodes[py_idx[2]][0] * dphi_dx_m[2] +
            w_nodes[py_idx[0]][1] * dphi_dy_m[0] + w_nodes[py_idx[1]][1] * dphi_dy_m[1] + w_nodes[py_idx[2]][1] * dphi_dy_m[2];

        // 2. Geometria su t^{n+1}
        const double x1_p = nodes_next[py_idx[0]][0], y1_p = nodes_next[py_idx[0]][1];
        const double x2_p = nodes_next[py_idx[1]][0], y2_p = nodes_next[py_idx[1]][1];
        const double x3_p = nodes_next[py_idx[2]][0], y3_p = nodes_next[py_idx[2]][1];

        const double det_j_p = (x2_p - x1_p) * (y3_p - y1_p) - (x3_p - x1_p) * (y2_p - y1_p);
        const double area_next = 0.5 * std::abs(det_j_p);
        const double inv_det_p = 1.0 / det_j_p;

        const std::array<double, 3> dphi_dx_p = {
            (y2_p - y3_p) * inv_det_p, (y3_p - y1_p) * inv_det_p, (y1_p - y2_p) * inv_det_p
        };
        const std::array<double, 3> dphi_dy_p = {
            (x3_p - x2_p) * inv_det_p, (x1_p - x3_p) * inv_det_p, (x2_p - x1_p) * inv_det_p
        };

        // 3. Geometria su t^n
        const double x1_n = nodes_n[py_idx[0]][0], y1_n = nodes_n[py_idx[0]][1];
        const double x2_n = nodes_n[py_idx[1]][0], y2_n = nodes_n[py_idx[1]][1];
        const double x3_n = nodes_n[py_idx[2]][0], y3_n = nodes_n[py_idx[2]][1];
        const double det_j_n = (x2_n - x1_n) * (y3_n - y1_n) - (x3_n - x1_n) * (y2_n - y1_n);
        const double area_n = 0.5 * std::abs(det_j_n);

        // Residuo geometrico elementare D-GCL
        const double r_elem = std::abs((area_next - area_n) / dt - area_mid * div_w);
        diag.element_gcl_residuals[e] = r_elem;
        diag.max_elem_gcl_residual = std::max(diag.max_elem_gcl_residual, r_elem);

        // 4. Operatore conservativo C_{mesh}^{n+1/2} = -C_{adv}^{w,n+1/2} - M_{div}^{w,n+1/2}
        std::array<std::array<double, 3>, 3> C_adv_e{{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
        for (std::size_t q = 0; q < n_q; ++q) {
            const double xi = q_pts[q][0];
            const double eta = q_pts[q][1];
            const std::array<double, 3> phi = {1.0 - xi - eta, xi, eta};
            // q_wts[q] in deal.II somma a 0.5, quindi 2 * q_wts[q] * area_mid e' il peso fisico
            const double jxw_mid = (2.0 * q_wts[q]) * area_mid;

            const double wx_q = phi[0] * w_nodes[py_idx[0]][0] + phi[1] * w_nodes[py_idx[1]][0] + phi[2] * w_nodes[py_idx[2]][0];
            const double wy_q = phi[0] * w_nodes[py_idx[0]][1] + phi[1] * w_nodes[py_idx[1]][1] + phi[2] * w_nodes[py_idx[2]][1];

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    const double w_dot_grad_j = wx_q * dphi_dx_m[j] + wy_q * dphi_dy_m[j];
                    C_adv_e[i][j] += jxw_mid * phi[i] * w_dot_grad_j;
                }
            }
        }

        // 5. Termini fisici su t^{n+1}
        std::array<std::array<double, 3>, 3> C_phys_e{{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
        std::array<std::array<double, 3>, 3> K_e{{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
        std::array<double, 3> F_e = {0.0, 0.0, 0.0};

        if (!pure_gcl) {
            for (std::size_t q = 0; q < n_q; ++q) {
                const double xi = q_pts[q][0];
                const double eta = q_pts[q][1];
                const std::array<double, 3> phi = {1.0 - xi - eta, xi, eta};
                const double jxw_next = (2.0 * q_wts[q]) * area_next;

                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        const double b_grad_j = b_[0] * dphi_dx_p[j] + b_[1] * dphi_dy_p[j];
                        C_phys_e[i][j] += jxw_next * phi[i] * b_grad_j;
                    }
                }

                if (pde_ != nullptr) {
                    const double xq = phi[0] * x1_p + phi[1] * x2_p + phi[2] * x3_p;
                    const double yq = phi[0] * y1_p + phi[1] * y2_p + phi[2] * y3_p;
                    const double f_val = pde_->evaluate_forcing(xq, yq, t_next);
                    for (int i = 0; i < 3; ++i) {
                        F_e[i] += jxw_next * f_val * phi[i];
                    }
                }
            }

            if (nu_ > 0.0) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        const double grad_dot = dphi_dx_p[i] * dphi_dx_p[j] + dphi_dy_p[i] * dphi_dy_p[j];
                        K_e[i][j] = nu_ * area_next * grad_dot;
                    }
                }
            }
        }

        for (int i = 0; i < 3; ++i) {
            F_vec[dofs[i]] += F_e[i];
            for (int j = 0; j < 3; ++j) {
                const double m_next_ij = area_next * kBaseRefMass[i][j];
                const double m_n_ij = area_n * kBaseRefMass[i][j];
                // Termine cinematico di dilatazione valutato sul midpoint X^{n+1/2} (non tautologico)
                const double m_div_w_ij = (area_mid * div_w) * kBaseRefMass[i][j];
                const double c_mesh_ij = -C_adv_e[i][j] - m_div_w_ij;
                const double a_ij = (1.0 / dt) * m_next_ij + c_mesh_ij + C_phys_e[i][j] + K_e[i][j];

                M_next_mat.add(dofs[i], dofs[j], m_next_ij);
                M_n_mat.add(dofs[i], dofs[j], m_n_ij);
                C_mesh_mat.add(dofs[i], dofs[j], c_mesh_ij);
                A_mat.add(dofs[i], dofs[j], a_ij);
            }
        }
    }

    // Diagnostica algebrica D-GCL: || ((M^{n+1} - M^n)/dt + C_mesh^{n+1/2}) * 1 ||_inf
    dealii::Vector<double> ones(n_nodes), tmp_next(n_nodes), tmp_n(n_nodes), gcl_vec(n_nodes);
    ones = 1.0;
    M_next_mat.vmult(tmp_next, ones);
    M_n_mat.vmult(tmp_n, ones);
    C_mesh_mat.vmult(gcl_vec, ones);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        gcl_vec[i] += (tmp_next[i] - tmp_n[i]) / dt;
    }
    diag.algebraic_gcl_residual_linf = gcl_vec.linfty_norm();

    // Diagnostica duale: || 1^T C_mesh^{n+1/2} ||_inf
    dealii::Vector<double> dual_vec(n_nodes);
    C_mesh_mat.Tvmult(dual_vec, ones);
    diag.dual_c_mesh_col_sum_linf = dual_vec.linfty_norm();

    // Assemblaggio termine noto: rhs = (1/dt) * M^n * u^n + F^{n+1}
    dealii::Vector<double> u_n_deal(n_nodes), rhs(n_nodes);
    for (std::size_t py_i = 0; py_i < n_nodes; ++py_i) {
        u_n_deal[mesh_next.py_node_to_dof(py_i)] = u_n[py_i];
    }
    M_n_mat.vmult(rhs, u_n_deal);
    diag.discrete_mass_n = rhs * ones;
    rhs *= (1.0 / dt);
    rhs += F_vec;

    dealii::Vector<double> u_next_deal(n_nodes);
    if (enforce_dirichlet) {
        std::map<dealii::types::global_dof_index, double> boundary_values;
        for (std::int64_t b_py : mesh_next.boundary_nodes()) {
            const auto idx = static_cast<std::size_t>(b_py);
            const auto dof = mesh_next.py_node_to_dof(idx);
            if (pure_gcl) {
                boundary_values[dof] = 1.0;
            } else if (pde_ != nullptr) {
                boundary_values[dof] = pde_->evaluate_solution(
                    nodes_next[idx][0], nodes_next[idx][1], t_next);
            } else {
                boundary_values[dof] = 0.0;
            }
        }
        dealii::MatrixTools::apply_boundary_values(
            boundary_values, A_mat, u_next_deal, rhs, false);
    }

    if (solver_params_.use_direct_solver) {
        dealii::SparseDirectUMFPACK direct_solver;
        direct_solver.initialize(A_mat);
        direct_solver.vmult(u_next_deal, rhs);
    } else {
        const double tol = std::max(solver_params_.abs_tol,
                                    solver_params_.rel_tol * std::max(rhs.l2_norm(), 1.0));
        dealii::SolverControl solver_control(
            static_cast<unsigned int>(solver_params_.max_iterations), tol);
        dealii::SolverGMRES<dealii::Vector<double>>::AdditionalData gmres_data(
            static_cast<unsigned int>(solver_params_.gmres_krylov_dim));
        dealii::SolverGMRES<dealii::Vector<double>> solver(solver_control, gmres_data);

        dealii::PreconditionSSOR<dealii::SparseMatrix<double>> preconditioner;
        preconditioner.initialize(A_mat, solver_params_.ssor_omega);
        solver.solve(A_mat, u_next_deal, rhs, preconditioner);
    }

    dealii::Vector<double> res_vec(n_nodes);
    A_mat.vmult(res_vec, u_next_deal);
    res_vec -= rhs;
    diag.linear_solver_residual = res_vec.l2_norm();

    dealii::Vector<double> m_next_u(n_nodes);
    M_next_mat.vmult(m_next_u, u_next_deal);
    diag.discrete_mass_next = m_next_u * ones;

    ALEStepResult out;
    out.u_next.resize(n_nodes);
    for (std::size_t py_i = 0; py_i < n_nodes; ++py_i) {
        out.u_next[py_i] = u_next_deal[mesh_next.py_node_to_dof(py_i)];
    }
    out.diagnostics = std::move(diag);
    return out;
}

ALEDiagnostics ALESolver2D::evaluate_gcl_diagnostics(
    const MeshState& mesh_n,
    const MeshState& mesh_next,
    double dt) const {
    const std::vector<double> ones(mesh_next.num_nodes(), 1.0);
    return step_ale(mesh_n, mesh_next, ones, dt, dt, true, true).diagnostics;
}

} // namespace gdm