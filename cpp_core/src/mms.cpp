#include "gdm/mms.hpp"
#include "gdm/ale_solver.hpp"
#include "gdm/fem_solver.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace gdm {

std::vector<MeshState::Coord2D> analytical_mesh_motion(
    const std::vector<MeshState::Coord2D>& nodes_0,
    double t,
    double T,
    double amplitude) {
    const double pi = std::numbers::pi;
    const double omega = 2.0 * pi / T;
    std::vector<MeshState::Coord2D> out(nodes_0.size());
    for (std::size_t i = 0; i < nodes_0.size(); ++i) {
        const double x = nodes_0[i][0];
        const double y = nodes_0[i][1];
        const double spatial = std::sin(pi * x) * std::sin(pi * y);
        const double dx = amplitude * spatial * std::sin(omega * t);
        const double dy = 0.5 * amplitude * spatial * (std::cos(omega * t) - 1.0);
        out[i] = {x + dx, y + dy};
    }
    return out;
}

std::vector<MeshState::Coord2D> analytical_mesh_velocity(
    const std::vector<MeshState::Coord2D>& nodes_0,
    double t,
    double T,
    double amplitude) {
    const double pi = std::numbers::pi;
    const double omega = 2.0 * pi / T;
    std::vector<MeshState::Coord2D> w(nodes_0.size());
    for (std::size_t i = 0; i < nodes_0.size(); ++i) {
        const double x = nodes_0[i][0];
        const double y = nodes_0[i][1];
        const double spatial = std::sin(pi * x) * std::sin(pi * y);
        const double wx = amplitude * spatial * omega * std::cos(omega * t);
        const double wy = -0.5 * amplitude * spatial * omega * std::sin(omega * t);
        w[i] = {wx, wy};
    }
    return w;
}

std::vector<SpatialConvergenceRow> run_fem_spatial_convergence_study(
    const std::vector<std::size_t>& grid_resolutions,
    const PDEParameters& pde_params,
    double t_final,
    double base_dt_at_n8) {
    if (grid_resolutions.empty()) {
        throw std::invalid_argument("run_fem_spatial_convergence_study: lista risoluzioni vuota.");
    }
    ManufacturedConvectionDiffusion2D pde(pde_params);
    std::vector<SpatialConvergenceRow> table;
    table.reserve(grid_resolutions.size());

    for (std::size_t idx = 0; idx < grid_resolutions.size(); ++idx) {
        const std::size_t n = grid_resolutions[idx];
        DomainParameters dom;
        dom.nx = n;
        dom.ny = n;
        MeshState mesh = MeshState::create_structured_mesh(dom);
        ConvectionDiffusionFEM2D solver(mesh, pde);

        // Scaling dt ~ h^2 per isolare il secondo ordine spaziale P1 senza saturazione Backward Euler
        const double scale = static_cast<double>(n) / 8.0;
        const double dt_raw = base_dt_at_n8 / (scale * scale);
        const auto n_steps = static_cast<std::size_t>(std::max(1.0, std::round(t_final / dt_raw)));
        const double dt = t_final / static_cast<double>(n_steps);

        std::vector<double> u = pde.evaluate_solution_on_nodes(mesh.nodes(), 0.0);
        for (std::size_t step = 1; step <= n_steps; ++step) {
            const double t_next = static_cast<double>(step) * dt;
            u = solver.step_backward_euler(u, dt, t_next);
        }

        const auto norms = compute_continuous_errors(mesh, u, pde, t_final);
        SpatialConvergenceRow row;
        row.n_subdivisions = n;
        row.h = 1.0 / static_cast<double>(n);
        row.rel_l2_error = norms.rel_l2_error;
        row.rel_h1_error = norms.rel_h1_seminorm_error;

        if (idx > 0) {
            const double log_h_ratio = std::log(table[idx - 1].h / row.h);
            row.order_l2 = std::log(table[idx - 1].rel_l2_error / row.rel_l2_error) / log_h_ratio;
            row.order_h1 = std::log(table[idx - 1].rel_h1_error / row.rel_h1_error) / log_h_ratio;
        }
        table.push_back(row);
    }
    return table;
}

std::vector<SpatialConvergenceRow> run_ale_spatial_convergence_study(
    const std::vector<std::size_t>& grid_resolutions,
    const PDEParameters& pde_params,
    double t_final,
    double base_dt_at_n8,
    double motion_amplitude) {
    if (grid_resolutions.empty()) {
        throw std::invalid_argument("run_ale_spatial_convergence_study: lista risoluzioni vuota.");
    }
    ManufacturedConvectionDiffusion2D pde(pde_params);
    ALESolver2D ale_solver(&pde);
    std::vector<SpatialConvergenceRow> table;
    table.reserve(grid_resolutions.size());

    for (std::size_t idx = 0; idx < grid_resolutions.size(); ++idx) {
        const std::size_t n = grid_resolutions[idx];
        DomainParameters dom;
        dom.nx = n;
        dom.ny = n;
        MeshState mesh_curr = MeshState::create_structured_mesh(dom);
        const auto nodes_0 = mesh_curr.nodes();

        const double scale = static_cast<double>(n) / 8.0;
        const double dt_raw = base_dt_at_n8 / (scale * scale);
        const auto n_steps = static_cast<std::size_t>(std::max(1.0, std::round(t_final / dt_raw)));
        const double dt = t_final / static_cast<double>(n_steps);

        std::vector<double> u = pde.evaluate_solution_on_nodes(nodes_0, 0.0);
        for (std::size_t step = 1; step <= n_steps; ++step) {
            const double t_next = static_cast<double>(step) * dt;
            const auto nodes_next = analytical_mesh_motion(nodes_0, t_next, pde_params.T, motion_amplitude);
            MeshState mesh_next = mesh_curr.cloned_with_coordinates(nodes_next, true);
            auto step_res = ale_solver.step_ale(mesh_curr, mesh_next, u, dt, t_next, false, true);
            u = std::move(step_res.u_next);
            mesh_curr = std::move(mesh_next);
        }

        const auto norms = compute_continuous_errors(mesh_curr, u, pde, t_final);
        SpatialConvergenceRow row;
        row.n_subdivisions = n;
        row.h = 1.0 / static_cast<double>(n);
        row.rel_l2_error = norms.rel_l2_error;
        row.rel_h1_error = norms.rel_h1_seminorm_error;

        if (idx > 0) {
            const double log_h_ratio = std::log(table[idx - 1].h / row.h);
            row.order_l2 = std::log(table[idx - 1].rel_l2_error / row.rel_l2_error) / log_h_ratio;
            row.order_h1 = std::log(table[idx - 1].rel_h1_error / row.rel_h1_error) / log_h_ratio;
        }
        table.push_back(row);
    }
    return table;
}

std::vector<TemporalConvergenceRow> run_ale_temporal_convergence_study(
    std::size_t grid_n,
    const std::vector<double>& dt_list,
    const PDEParameters& pde_params,
    double t_final,
    double motion_amplitude) {
    if (dt_list.size() < 2) {
        throw std::invalid_argument("run_ale_temporal_convergence_study: servono almeno 2 passi dt.");
    }
    ManufacturedConvectionDiffusion2D pde(pde_params);
    ALESolver2D ale_solver(&pde);

    DomainParameters dom;
    dom.nx = grid_n;
    dom.ny = grid_n;
    MeshState mesh_0 = MeshState::create_structured_mesh(dom);
    const auto nodes_0 = mesh_0.nodes();
    const auto nodes_T = analytical_mesh_motion(nodes_0, t_final, pde_params.T, motion_amplitude);
    const MeshState mesh_T = mesh_0.cloned_with_coordinates(nodes_T, true);

    auto run_for_dt = [&](double dt) -> std::vector<double> {
        const auto n_steps = static_cast<std::size_t>(std::round(t_final / dt));
        MeshState mesh_curr = mesh_0;
        std::vector<double> u = pde.evaluate_solution_on_nodes(nodes_0, 0.0);
        for (std::size_t step = 1; step <= n_steps; ++step) {
            const double t_next = static_cast<double>(step) * dt;
            const auto nodes_next = analytical_mesh_motion(nodes_0, t_next, pde_params.T, motion_amplitude);
            MeshState mesh_next = mesh_curr.cloned_with_coordinates(nodes_next, true);
            u = ale_solver.step_ale(mesh_curr, mesh_next, u, dt, t_next, false, true).u_next;
            mesh_curr = std::move(mesh_next);
        }
        return u;
    };

    // Soluzione di riferimento temporale con passo dimezzato rispetto al piu' fine
    const double dt_ref = dt_list.back() * 0.25;
    const std::vector<double> u_ref = run_for_dt(dt_ref);

    std::vector<TemporalConvergenceRow> rows;
    rows.reserve(dt_list.size());
    for (std::size_t i = 0; i < dt_list.size(); ++i) {
        const double dt = dt_list[i];
        const std::vector<double> u_dt = run_for_dt(dt);
        // Norma L2(Omega(T)) indotta dalla matrice di massa finale M(T)
        const double err = compute_mass_matrix_l2_difference(mesh_T, u_dt, u_ref);

        TemporalConvergenceRow r;
        r.dt = dt;
        r.cauchy_l2_error = err;
        if (i > 0) {
            r.order_l2 = std::log(rows[i - 1].cauchy_l2_error / r.cauchy_l2_error) /
                         std::log(rows[i - 1].dt / r.dt);
        }
        rows.push_back(r);
    }
    return rows;
}

} // namespace gdm