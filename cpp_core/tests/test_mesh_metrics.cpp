#include "test_utils.hpp"
#include "gdm/mesh_metrics.hpp"
#include "gdm/mesh_state.hpp"

#include <cmath>
#include <vector>

int main() {
    using namespace gdm;

    // 1. Triangolo equilatero di lato 1 orientato positivamente
    const std::vector<MeshState::Coord2D> eq_nodes = {
        {0.0, 0.0}, {1.0, 0.0}, {0.5, std::sqrt(3.0) * 0.5}
    };
    const std::vector<MeshState::CellConn> eq_cells = {{0, 1, 2}};
    const auto m_eq = compute_mesh_metrics(eq_nodes, eq_cells);

    GDM_CHECK(m_eq.num_inverted == 0);
    GDM_CHECK(m_eq.num_degenerate == 0);
    GDM_CHECK(std::abs(m_eq.jacobians[0] - std::sqrt(3.0) * 0.5) < 1.0e-14);
    GDM_CHECK(std::abs(m_eq.escobar_quality[0] - 1.0) < 1.0e-14);
    GDM_CHECK(std::abs(m_eq.aspect_ratio[0] - 1.0) < 1.0e-12);

    // 2. Triangolo con orientamento invertito (scambio vertici 1 e 2)
    // Verifica che il determinante sia negativo e non mascherato da valore assoluto
    const std::vector<MeshState::CellConn> inv_cells = {{0, 2, 1}};
    const auto m_inv = compute_mesh_metrics(eq_nodes, inv_cells);

    GDM_CHECK(m_inv.jacobians[0] < 0.0);
    GDM_CHECK(m_inv.areas[0] < 0.0);
    GDM_CHECK(m_inv.inverted[0] == true);
    GDM_CHECK(m_inv.num_inverted == 1);
    GDM_CHECK(m_inv.escobar_quality[0] == 0.0);

    // 3. Mesh strutturata 32x32 su [0,1]^2: J_e = 1 / 1024 = 9.765625e-4
    DomainParameters dom;
    dom.nx = 32;
    dom.ny = 32;
    MeshState mesh = MeshState::create_structured_mesh(dom);
    const auto m_grid = compute_mesh_metrics(mesh);

    GDM_CHECK(m_grid.num_inverted == 0);
    GDM_CHECK(std::abs(m_grid.min_jacobian - 1.0 / 1024.0) < 1.0e-15);
    GDM_CHECK(std::abs(m_grid.max_jacobian - 1.0 / 1024.0) < 1.0e-15);

    return 0;
}