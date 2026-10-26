#include "gdm/remap.hpp"
#include "gdm/ale_solver.hpp"
#include "gdm/pde_model.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gdm {

namespace {

struct PreparedSourceTriangles {
    struct TriData {
        double x1, y1, x2, y2, x3, y3;
        double min_x, max_x, min_y, max_y;
        double cx, cy;
        double inv_det;
        std::size_t i1, i2, i3;
    };

    std::vector<TriData> tris;

    explicit PreparedSourceTriangles(const MeshState& mesh) {
        const auto& nodes = mesh.nodes();
        const auto& cells = mesh.cells();
        tris.resize(cells.size());
        for (std::size_t e = 0; e < cells.size(); ++e) {
            const auto i1 = static_cast<std::size_t>(cells[e][0]);
            const auto i2 = static_cast<std::size_t>(cells[e][1]);
            const auto i3 = static_cast<std::size_t>(cells[e][2]);

            const double x1 = nodes[i1][0], y1 = nodes[i1][1];
            const double x2 = nodes[i2][0], y2 = nodes[i2][1];
            const double x3 = nodes[i3][0], y3 = nodes[i3][1];
            const double det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);

            tris[e] = {
                x1, y1, x2, y2, x3, y3,
                std::min({x1, x2, x3}), std::max({x1, x2, x3}),
                std::min({y1, y2, y3}), std::max({y1, y2, y3}),
                (x1 + x2 + x3) / 3.0, (y1 + y2 + y3) / 3.0,
                1.0 / det,
                i1, i2, i3
            };
        }
    }

    double evaluate_at(double px, double py,
                       const std::vector<double>& old_u,
                       std::size_t& fallback_counter) const {
        constexpr double tol_bary = -1.0e-11;
        double best_min_lambda = -std::numeric_limits<double>::infinity();
        double best_val = 0.0;
        bool found_inside = false;

        for (const auto& t : tris) {
            if (px < t.min_x - 1.0e-10 || px > t.max_x + 1.0e-10 ||
                py < t.min_y - 1.0e-10 || py > t.max_y + 1.0e-10) {
                continue;
            }
            const double l2 = ((px - t.x1) * (t.y3 - t.y1) - (t.x3 - t.x1) * (py - t.y1)) * t.inv_det;
            const double l3 = ((t.x2 - t.x1) * (py - t.y1) - (px - t.x1) * (t.y2 - t.y1)) * t.inv_det;
            const double l1 = 1.0 - l2 - l3;
            const double min_l = std::min({l1, l2, l3});

            if (min_l >= tol_bary) {
                return l1 * old_u[t.i1] + l2 * old_u[t.i2] + l3 * old_u[t.i3];
            }
            if (min_l > best_min_lambda) {
                best_min_lambda = min_l;
                best_val = l1 * old_u[t.i1] + l2 * old_u[t.i2] + l3 * old_u[t.i3];
                found_inside = true;
            }
        }

        if (found_inside && best_min_lambda >= -1.0e-4) {
            return best_val;
        }

        // Fallback deterministico sul triangolo a minore violazione baricentrica su tutta la griglia
        ++fallback_counter;
        for (const auto& t : tris) {
            const double l2 = ((px - t.x1) * (t.y3 - t.y1) - (t.x3 - t.x1) * (py - t.y1)) * t.inv_det;
            const double l3 = ((t.x2 - t.x1) * (py - t.y1) - (px - t.x1) * (t.y2 - t.y1)) * t.inv_det;
            const double l1 = 1.0 - l2 - l3;
            const double min_l = std::min({l1, l2, l3});
            if (min_l > best_min_lambda) {
                best_min_lambda = min_l;
                best_val = l1 * old_u[t.i1] + l2 * old_u[t.i2] + l3 * old_u[t.i3];
            }
        }
        return best_val;
    }
};

} // namespace

RemapDiagnostics remap_p1_solution(
    const MeshState& old_mesh,
    const std::vector<double>& old_u,
    const MeshState& new_mesh) {
    if (old_u.size() != old_mesh.num_nodes()) {
        throw std::invalid_argument("remap_p1_solution: dimensione old_u non valida.");
    }

    const auto t0 = std::chrono::steady_clock::now();
    PreparedSourceTriangles locator(old_mesh);

    const auto& new_nodes = new_mesh.nodes();
    RemapDiagnostics out;
    out.remapped_field.resize(new_nodes.size());

    for (std::size_t i = 0; i < new_nodes.size(); ++i) {
        out.remapped_field[i] = locator.evaluate_at(
            new_nodes[i][0], new_nodes[i][1], old_u, out.fallback_nearest_cell_count);
    }

    const auto t1 = std::chrono::steady_clock::now();
    out.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out.mass_before = ALESolver2D::compute_discrete_mass(old_mesh, old_u);
    out.mass_after = ALESolver2D::compute_discrete_mass(new_mesh, out.remapped_field);
    out.relative_mass_drift = std::abs(out.mass_after - out.mass_before) /
                              std::max(std::abs(out.mass_before), 1.0e-14);
    return out;
}

RemapDiagnostics remap_l2_projection(
    const MeshState& old_mesh,
    const std::vector<double>& old_u,
    const MeshState& new_mesh,
    const LinearSolverParameters& solver_params) {
    solver_params.validate();
    if (old_u.size() != old_mesh.num_nodes()) {
        throw std::invalid_argument("remap_l2_projection: dimensione old_u non valida.");
    }

    const auto t0 = std::chrono::steady_clock::now();
    PreparedSourceTriangles locator(old_mesh);

    const std::size_t n_nodes = new_mesh.num_nodes();
    const auto& new_nodes = new_mesh.nodes();
    const auto& new_cells = new_mesh.cells();

    const auto& dof_handler = new_mesh.dof_handler();
    dealii::DynamicSparsityPattern dsp(n_nodes, n_nodes);
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);
    dealii::SparsityPattern sparsity;
    sparsity.copy_from(dsp);

    dealii::SparseMatrix<double> M_new(sparsity);
    dealii::Vector<double> rhs(n_nodes);

    const auto quad = create_dunavant_degree5_quadrature();
    const auto& q_pts = quad.get_points();
    const auto& q_wts = quad.get_weights();
    const std::size_t n_q = q_pts.size();

    constexpr std::array<std::array<double, 3>, 3> base_M = {{
        {2.0 / 12.0, 1.0 / 12.0, 1.0 / 12.0},
        {1.0 / 12.0, 2.0 / 12.0, 1.0 / 12.0},
        {1.0 / 12.0, 1.0 / 12.0, 2.0 / 12.0}
    }};

    RemapDiagnostics out;

    for (const auto& c : new_cells) {
        const std::array<std::size_t, 3> py_idx = {
            static_cast<std::size_t>(c[0]),
            static_cast<std::size_t>(c[1]),
            static_cast<std::size_t>(c[2])
        };
        const std::array<dealii::types::global_dof_index, 3> dofs = {
            new_mesh.py_node_to_dof(py_idx[0]),
            new_mesh.py_node_to_dof(py_idx[1]),
            new_mesh.py_node_to_dof(py_idx[2])
        };

        const double x1 = new_nodes[py_idx[0]][0], y1 = new_nodes[py_idx[0]][1];
        const double x2 = new_nodes[py_idx[1]][0], y2 = new_nodes[py_idx[1]][1];
        const double x3 = new_nodes[py_idx[2]][0], y3 = new_nodes[py_idx[2]][1];

        const double det_j = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
        const double area = 0.5 * std::abs(det_j);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                M_new.add(dofs[i], dofs[j], area * base_M[i][j]);
            }
        }

        for (std::size_t q = 0; q < n_q; ++q) {
            const double xi = q_pts[q][0];
            const double eta = q_pts[q][1];
            const std::array<double, 3> phi = {1.0 - xi - eta, xi, eta};

            const double xq = phi[0] * x1 + phi[1] * x2 + phi[2] * x3;
            const double yq = phi[0] * y1 + phi[1] * y2 + phi[2] * y3;
            const double u_old_q = locator.evaluate_at(
                xq, yq, old_u, out.fallback_nearest_cell_count);
            const double jxw = (2.0 * q_wts[q]) * area;

            for (int i = 0; i < 3; ++i) {
                rhs[dofs[i]] += jxw * u_old_q * phi[i];
            }
        }
    }

    dealii::Vector<double> sol_deal(n_nodes);
    const double tol = std::max(solver_params.abs_tol,
                                solver_params.rel_tol * std::max(rhs.l2_norm(), 1.0));
    dealii::SolverControl solver_control(
        static_cast<unsigned int>(solver_params.max_iterations), tol);
    dealii::SolverCG<dealii::Vector<double>> cg_solver(solver_control);
    dealii::PreconditionSSOR<dealii::SparseMatrix<double>> preconditioner;
    preconditioner.initialize(M_new, solver_params.ssor_omega);
    cg_solver.solve(M_new, sol_deal, rhs, preconditioner);

    out.remapped_field.resize(n_nodes);
    for (std::size_t py_i = 0; py_i < n_nodes; ++py_i) {
        out.remapped_field[py_i] = sol_deal[new_mesh.py_node_to_dof(py_i)];
    }

    const auto t1 = std::chrono::steady_clock::now();
    out.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out.mass_before = ALESolver2D::compute_discrete_mass(old_mesh, old_u);
    out.mass_after = ALESolver2D::compute_discrete_mass(new_mesh, out.remapped_field);
    out.relative_mass_drift = std::abs(out.mass_after - out.mass_before) /
                              std::max(std::abs(out.mass_before), 1.0e-14);
    return out;
}

} // namespace gdm