#include "test_utils.hpp"
#include "gdm/ale_solver.hpp"
#include "gdm/diagnostics.hpp"
#include "gdm/fem_solver.hpp"
#include "gdm/mms.hpp"

#include <cmath>
#include <vector>

int main() {
    using namespace gdm;

    PDEParameters pde_params;
    ManufacturedConvectionDiffusion2D pde(pde_params);

    // TEST C3: Verifica su singolo passo ALE contro soluzione manifatturata e passo su griglia fissa
    {
        DomainParameters dom;
        dom.nx = 32;
        dom.ny = 32;
        MeshState mesh_0 = MeshState::create_structured_mesh(dom);
        const double dt = 0.005;
        const auto u_0 = pde.evaluate_solution_on_nodes(mesh_0.nodes(), 0.0);

        ConvectionDiffusionFEM2D fem_fixed(mesh_0, pde);
        const auto u_fixed = fem_fixed.step_backward_euler(u_0, dt, dt);
        const auto err_fixed = compute_continuous_errors(mesh_0, u_fixed, pde, dt);

        ALESolver2D ale(&pde);
        const auto nodes_1 = analytical_mesh_motion(mesh_0.nodes(), dt, pde_params.T, 0.02);
        const MeshState mesh_1 = mesh_0.cloned_with_coordinates(nodes_1, true);
        const auto ale_res = ale.step_ale(mesh_0, mesh_1, u_0, dt, dt, false, true);
        const auto err_ale = compute_continuous_errors(mesh_1, ale_res.u_next, pde, dt);

        GDM_CHECK(err_fixed.rel_l2_error < 0.05);
        GDM_CHECK(err_ale.rel_l2_error < 0.05);
        GDM_CHECK(std::abs(err_ale.rel_l2_error - err_fixed.rel_l2_error) < 0.005);
    }

    // TEST C4: Convergenza spaziale ALE su mesh deformata (dt scalato ~ h^2)
    {
        const std::vector<std::size_t> grids = {8, 16, 32, 64};
        const auto spatial_rows = run_ale_spatial_convergence_study(
            grids, pde_params, 0.05, 1.0e-3, 0.02);

        GDM_CHECK(spatial_rows.size() == 4);
        GDM_CHECK(spatial_rows.back().order_l2 > 1.85);
        GDM_CHECK(spatial_rows.back().order_h1 > 0.90);
    }

    // TEST C5: Convergenza temporale ALE Backward Euler in norma L2(Omega(T)) indotta da M(T)
    {
        const std::vector<double> dt_list = {0.01, 0.005, 0.0025};
        const auto temp_rows = run_ale_temporal_convergence_study(
            32, dt_list, pde_params, 0.1, 0.02);

        GDM_CHECK(temp_rows.size() == 3);
        GDM_CHECK(temp_rows[1].order_l2 > 0.90);
        GDM_CHECK(temp_rows[2].order_l2 > 0.95);
    }

    return 0;
}