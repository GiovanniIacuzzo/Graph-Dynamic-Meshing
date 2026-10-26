#include "test_utils.hpp"
#include "gdm/diagnostics.hpp"
#include "gdm/fem_solver.hpp"
#include "gdm/mesh_state.hpp"
#include "gdm/pde_model.hpp"

#include <cmath>

int main() {
    using namespace gdm;

    PDEParameters pde_params;
    ManufacturedConvectionDiffusion2D pde(pde_params);

    // 1. Verifica derivate analitiche del modello PDE contro differenze finite centrate
    const double x = 0.31;
    const double y = 0.27;
    const double t = 0.12;
    const double h_fd = 1.0e-6;

    const auto ev = pde.evaluate_exact(x, y, t);
    const double ux_fd = (pde.evaluate_solution(x + h_fd, y, t) - pde.evaluate_solution(x - h_fd, y, t)) / (2.0 * h_fd);
    const double uy_fd = (pde.evaluate_solution(x, y + h_fd, t) - pde.evaluate_solution(x, y - h_fd, t)) / (2.0 * h_fd);
    const double ut_fd = (pde.evaluate_solution(x, y, t + h_fd) - pde.evaluate_solution(x, y, t - h_fd)) / (2.0 * h_fd);
    const double uxx_fd = (pde.evaluate_solution(x + h_fd, y, t) - 2.0 * ev.u + pde.evaluate_solution(x - h_fd, y, t)) / (h_fd * h_fd);
    const double uyy_fd = (pde.evaluate_solution(x, y + h_fd, t) - 2.0 * ev.u + pde.evaluate_solution(x, y - h_fd, t)) / (h_fd * h_fd);

    GDM_CHECK(std::abs(ev.ux - ux_fd) < 1.0e-7);
    GDM_CHECK(std::abs(ev.uy - uy_fd) < 1.0e-7);
    GDM_CHECK(std::abs(ev.ut - ut_fd) < 1.0e-7);
    GDM_CHECK(std::abs(ev.uxx - uxx_fd) < 1.0e-4);
    GDM_CHECK(std::abs(ev.uyy - uyy_fd) < 1.0e-4);

    // 2. Verifica di un passo Backward Euler FEM P1 su griglia 32x32
    DomainParameters dom;
    dom.nx = 32;
    dom.ny = 32;
    MeshState mesh = MeshState::create_structured_mesh(dom);
    ConvectionDiffusionFEM2D fem(mesh, pde);

    const double dt = 0.005;
    const auto u_0 = pde.evaluate_solution_on_nodes(mesh.nodes(), 0.0);
    double lin_res = 1.0;
    const auto u_1 = fem.step_backward_euler(u_0, dt, dt, &lin_res);

    const auto norms = compute_continuous_errors(mesh, u_1, pde, dt);
    GDM_CHECK(lin_res < 1.0e-10);
    GDM_CHECK(norms.rel_l2_error < 0.05);

    return 0;
}