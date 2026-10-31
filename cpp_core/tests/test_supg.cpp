#include "test_utils.hpp"
#include "gdm/ale_solver.hpp"
#include "gdm/diagnostics.hpp"
#include "gdm/fem_solver.hpp"
#include "gdm/mesh_state.hpp"
#include "gdm/pde_model.hpp"
#include "gdm/stabilization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

int main() {
    using namespace gdm;

    PDEParameters pde_params;
    pde_params.sigma = 0.12;
    ManufacturedConvectionDiffusion2D pde(pde_params);

    // Pointwise manufactured residual consistency: the analytical derivatives
    // and forcing must satisfy the declared strong PDE to floating-point accuracy.
    const auto exact = pde.evaluate_exact(0.237, 0.419, 0.173);
    const auto velocity = pde.velocity();
    const double strong_residual = exact.ut + velocity[0] * exact.ux + velocity[1] * exact.uy -
                                   pde.nu() * (exact.uxx + exact.uyy) - exact.f;
    GDM_CHECK(std::abs(strong_residual) < 2.0e-14);

    DomainParameters domain;
    domain.nx = 12;
    domain.ny = 12;
    MeshState mesh = MeshState::create_structured_mesh(domain);
    const double dt = 0.0025;
    const auto u0 = pde.evaluate_solution_on_nodes(mesh.nodes(), 0.0);

    StabilizationParameters galerkin_params;
    galerkin_params.use_supg = false;
    StabilizationParameters supg_params;
    supg_params.use_supg = true;

    // Fixed-grid SUPG assembly must be finite and yield a valid algebraic solve.
    LinearSolverParameters stage_e_solver_params;
    stage_e_solver_params.right_preconditioning = true;
    stage_e_solver_params.ale_rhs_norm_floor = 0.0;
    stage_e_solver_params.validate();
    // Keep legacy defaults intact for earlier campaigns while exercising the
    // explicit Stage-E right-preconditioned residual criterion here.
    GDM_CHECK(!LinearSolverParameters{}.right_preconditioning);
    GDM_CHECK(LinearSolverParameters{}.ale_rhs_norm_floor == 1.0);
    ConvectionDiffusionFEM2D fem_galerkin(mesh, pde, stage_e_solver_params, galerkin_params);
    ConvectionDiffusionFEM2D fem_supg(mesh, pde, stage_e_solver_params, supg_params);
    double residual_g = 0.0, residual_s = 0.0;
    double rhs_g = 0.0, rhs_s = 0.0, tol_g = 0.0, tol_s = 0.0;
    const auto u_g = fem_galerkin.step_backward_euler(u0, dt, dt, &residual_g, nullptr, nullptr, &rhs_g, &tol_g);
    const auto u_s = fem_supg.step_backward_euler(u0, dt, dt, &residual_s, nullptr, nullptr, &rhs_s, &tol_s);
    const auto err_g = compute_continuous_errors(mesh, u_g, pde, dt);
    const auto err_s = compute_continuous_errors(mesh, u_s, pde, dt);
    GDM_CHECK(std::isfinite(err_g.rel_l2_error));
    GDM_CHECK(std::isfinite(err_s.rel_l2_error));
    GDM_CHECK(tol_g > 0.0 && tol_s > 0.0);
    GDM_CHECK(residual_g <= 10.0 * tol_g);
    GDM_CHECK(residual_s <= 10.0 * tol_s);

    // On a stationary mesh, the conservative ALE residual reduces exactly to
    // the fixed-grid backward-Euler residual. This checks both SUPG paths.
    ALESolver2D ale_galerkin(&pde, pde.nu(), pde.velocity()[0], pde.velocity()[1],
                             stage_e_solver_params, galerkin_params);
    ALESolver2D ale_supg(&pde, pde.nu(), pde.velocity()[0], pde.velocity()[1],
                         stage_e_solver_params, supg_params);
    const auto ale_g = ale_galerkin.step_ale(mesh, mesh, u0, dt, dt, false, true);
    const auto ale_s = ale_supg.step_ale(mesh, mesh, u0, dt, dt, false, true);
    GDM_CHECK(compute_mass_matrix_l2_difference(mesh, u_g, ale_g.u_next) < 1.0e-9);
    GDM_CHECK(compute_mass_matrix_l2_difference(mesh, u_s, ale_s.u_next) < 1.0e-9);

    // If the mesh translates with velocity w=b, beta=b-w=0 and the SUPG
    // streamline test vanishes. Galerkin and SUPG ALE solutions must coincide.
    constexpr double bx = 0.4;
    constexpr double by = 0.2;
    std::vector<MeshState::Coord2D> translated = mesh.nodes();
    for (auto& point : translated) {
        point[0] += bx * dt;
        point[1] += by * dt;
    }
    MeshState translated_mesh = mesh.cloned_with_coordinates(translated, false);
    std::vector<double> smooth(mesh.num_nodes());
    for (std::size_t i = 0; i < mesh.num_nodes(); ++i) {
        const double x = mesh.nodes()[i][0] - 0.5;
        const double y = mesh.nodes()[i][1] - 0.5;
        smooth[i] = std::exp(-40.0 * (x * x + y * y));
    }
    ALESolver2D zero_beta_g(nullptr, 1.0e-3, bx, by, stage_e_solver_params, galerkin_params);
    ALESolver2D zero_beta_s(nullptr, 1.0e-3, bx, by, stage_e_solver_params, supg_params);
    const auto zero_g = zero_beta_g.step_ale(mesh, translated_mesh, smooth, dt, dt, false, false);
    const auto zero_s = zero_beta_s.step_ale(mesh, translated_mesh, smooth, dt, dt, false, false);
    GDM_CHECK(compute_mass_matrix_l2_difference(translated_mesh, zero_g.u_next, zero_s.u_next) < 1.0e-10);

    // Parameter-limit checks: tau remains finite and decreases in the
    // diffusion-dominated limit; the zero-speed streamline term is inactive.
    const std::array<std::array<double, 2>, 3> grads = {{{-1.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}}};
    const std::array<std::array<double, 2>, 3> vertices = {{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}};
    const std::array<double, 2> beta = {0.4, 0.2};
    const double h_beta = directional_element_length(grads, beta, vertices);
    const double tau_adv = supg_tau(dt, std::hypot(beta[0], beta[1]), h_beta, 1.0e-3, supg_params);
    const double tau_diff = supg_tau(dt, std::hypot(beta[0], beta[1]), h_beta, 1.0e3, supg_params);
    const double tau_zero = supg_tau(dt, 0.0, h_beta, 1.0e-3, supg_params);
    GDM_CHECK(std::isfinite(tau_adv) && tau_adv > 0.0);
    GDM_CHECK(std::isfinite(tau_diff) && tau_diff > 0.0 && tau_diff < tau_adv);
    GDM_CHECK(std::isfinite(tau_zero) && tau_zero > 0.0);

    // Constant-field consistency with zero physical and relative velocities.
    // The fully discrete residual is exactly zero and SUPG must not perturb it.
    ALESolver2D constant_g(nullptr, 0.0, 0.0, 0.0,
                           LinearSolverParameters{}, galerkin_params);
    ALESolver2D constant_s(nullptr, 0.0, 0.0, 0.0,
                           LinearSolverParameters{}, supg_params);
    const std::vector<double> constant_field(mesh.num_nodes(), 1.0);
    const auto constant_g_result = constant_g.step_ale(
        mesh, mesh, constant_field, dt, dt, false, false);
    const auto constant_s_result = constant_s.step_ale(
        mesh, mesh, constant_field, dt, dt, false, false);
    GDM_CHECK(compute_mass_matrix_l2_difference(mesh, constant_field,
                                                constant_g_result.u_next) < 1.0e-9);
    GDM_CHECK(compute_mass_matrix_l2_difference(mesh, constant_field,
                                                constant_s_result.u_next) < 1.0e-9);

    // Endpoint-valid meshes may nevertheless invert along the straight nodal
    // path. The ALE solver must reject such a trajectory before assembly.
    const std::vector<MeshState::Coord2D> tri_nodes = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
    const std::vector<MeshState::CellConn> tri_cells = {{0, 1, 2}};
    const std::vector<std::int64_t> tri_boundary = {0, 1, 2};
    MeshState tri_mesh(tri_nodes, tri_cells, tri_boundary);
    const std::vector<MeshState::Coord2D> reflected_nodes = {{1.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}};
    MeshState reflected_mesh(reflected_nodes, tri_cells, tri_boundary);
    bool rejected_interior_inversion = false;
    try {
        (void)zero_beta_g.step_ale(tri_mesh, reflected_mesh, {1.0, 1.0, 1.0}, dt, dt, true, true);
    } catch (const std::runtime_error&) {
        rejected_interior_inversion = true;
    }
    GDM_CHECK(rejected_interior_inversion);

    return 0;
}
