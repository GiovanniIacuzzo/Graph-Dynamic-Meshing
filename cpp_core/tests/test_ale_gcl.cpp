#include "test_utils.hpp"
#include "gdm/ale_solver.hpp"
#include "gdm/diagnostics.hpp"
#include "gdm/mesh_state.hpp"
#include "gdm/mms.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

int main() {
    using namespace gdm;

    DomainParameters dom;
    dom.nx = 32;
    dom.ny = 32;
    const MeshState mesh_0 = MeshState::create_structured_mesh(dom);
    const auto nodes_0 = mesh_0.nodes();

    const double T = 0.5;
    const double dt = 0.005;
    const double A = 0.02;

    // TEST C1: Verifica cinematica della velocita' di mesh al punto medio
    {
        const double t_n = 0.10;
        const auto nodes_n = analytical_mesh_motion(nodes_0, t_n, T, A);
        const auto nodes_next = analytical_mesh_motion(nodes_0, t_n + dt, T, A);
        const auto w_disc = ALESolver2D::compute_mesh_velocity(nodes_n, nodes_next, dt);
        const auto w_exact = analytical_mesh_velocity(nodes_0, t_n + 0.5 * dt, T, A);

        double diff_sq = 0.0;
        double ref_sq = 0.0;
        for (std::size_t i = 0; i < nodes_0.size(); ++i) {
            const double dx = w_disc[i][0] - w_exact[i][0];
            const double dy = w_disc[i][1] - w_exact[i][1];
            diff_sq += dx * dx + dy * dy;
            ref_sq += w_exact[i][0] * w_exact[i][0] + w_exact[i][1] * w_exact[i][1];
        }
        const double rel_l2_w = std::sqrt(diff_sq / ref_sq);
        GDM_CHECK(rel_l2_w < 5.0e-3);
    }

    // TEST C2: Verifica indipendente di D-GCL elementare, identita' algebrica e preservazione u = 1
    ALESolver2D ale_pure(nullptr, 0.0, 0.0, 0.0);
    {
        MeshState curr_mesh = mesh_0;
        std::vector<double> u_curr(mesh_0.num_nodes(), 1.0);

        double max_elem_res = 0.0;
        double max_alg_res = 0.0;
        double max_fs_err = 0.0;

        for (int step = 1; step <= 100; ++step) {
            const double t_next = static_cast<double>(step) * dt;
            const auto nodes_next = analytical_mesh_motion(nodes_0, t_next, T, A);
            MeshState next_mesh = curr_mesh.cloned_with_coordinates(nodes_next, true);

            const auto res = ale_pure.step_ale(curr_mesh, next_mesh, u_curr, dt, t_next, true, true);
            u_curr = res.u_next;
            curr_mesh = std::move(next_mesh);

            max_elem_res = std::max(max_elem_res, res.diagnostics.max_elem_gcl_residual);
            max_alg_res = std::max(max_alg_res, res.diagnostics.algebraic_gcl_residual_linf);
            max_fs_err = std::max(max_fs_err, compute_free_stream_error_linf(u_curr, 1.0));
        }

        GDM_CHECK(max_elem_res < 1.0e-14);
        GDM_CHECK(max_alg_res < 1.0e-14);
        GDM_CHECK(max_fs_err < 1.0e-10);
    }

    // TEST C2b: Conservazione globale della massa discreta per un campo non uniforme senza forzatura Dirichlet
    {
        MeshState curr_mesh = mesh_0;
        std::vector<double> u_curr(mesh_0.num_nodes());
        for (std::size_t i = 0; i < nodes_0.size(); ++i) {
            const double dx = nodes_0[i][0] - 0.5;
            const double dy = nodes_0[i][1] - 0.5;
            u_curr[i] = std::exp(-(dx * dx + dy * dy) / (2.0 * 0.08 * 0.08));
        }

        const double m_0 = ALESolver2D::compute_discrete_mass(curr_mesh, u_curr);
        double max_dual_norm = 0.0;
        double max_mass_drift = 0.0;

        for (int step = 1; step <= 100; ++step) {
            const double t_next = static_cast<double>(step) * dt;
            const auto nodes_next = analytical_mesh_motion(nodes_0, t_next, T, A);
            MeshState next_mesh = curr_mesh.cloned_with_coordinates(nodes_next, true);

            const auto res = ale_pure.step_ale(curr_mesh, next_mesh, u_curr, dt, t_next, true, false);
            u_curr = res.u_next;
            curr_mesh = std::move(next_mesh);

            max_dual_norm = std::max(max_dual_norm, res.diagnostics.dual_c_mesh_col_sum_linf);
            max_mass_drift = std::max(
                max_mass_drift,
                compute_relative_mass_drift(m_0, res.diagnostics.discrete_mass_next));
        }

        GDM_CHECK(max_dual_norm < 1.0e-14);
        GDM_CHECK(max_mass_drift < 1.0e-12);
    }

    return 0;
}