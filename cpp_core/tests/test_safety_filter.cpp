#include "test_utils.hpp"
#include "gdm/safety_filter.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
    using namespace gdm;

    SafetyParameters params;
    params.epsilon = 1.0e-4;
    params.safety_margin = 1.0e-6;
    params.tol_deg = 1.0e-12;

    const std::vector<MeshState::Coord2D> valid_nodes = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {0.25, 0.25}
    };
    const std::vector<MeshState::CellConn> cells = {
        {0, 1, 3}, {1, 2, 3}, {2, 0, 3}
    };
    const std::vector<std::int64_t> boundary = {0, 1, 2};

    // Spostamento nullo su mesh valida.
    const std::vector<MeshState::Coord2D> zero_disp(valid_nodes.size(), {0.0, 0.0});
    const auto res_zero = apply_safety_filter(valid_nodes, cells, zero_disp, boundary, params);
    GDM_CHECK(res_zero.is_safe);
    GDM_CHECK(std::abs(res_zero.alpha_global - 1.0) < 1.0e-15);
    GDM_CHECK(!res_zero.filter_activated);
    GDM_CHECK(std::abs(res_zero.min_jacobian_after - res_zero.min_jacobian_before) < 1.0e-15);

    // Una mesh iniziale invertita viene rifiutata, coerentemente con Python safety.py.
    const std::vector<MeshState::Coord2D> inverted_nodes = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {-0.5, -0.5}
    };
    bool rejected_inverted_initial_mesh = false;
    try {
        (void)apply_safety_filter(inverted_nodes, cells, zero_disp, boundary, params);
    } catch (const std::invalid_argument&) {
        rejected_inverted_initial_mesh = true;
    }
    GDM_CHECK(rejected_inverted_initial_mesh);

    // Piccolo spostamento ammesso senza rilassamento.
    const std::vector<MeshState::Coord2D> small_disp = {
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.02, 0.02}
    };
    const auto res_adm = apply_safety_filter(valid_nodes, cells, small_disp, boundary, params);
    GDM_CHECK(res_adm.is_safe);
    GDM_CHECK(std::abs(res_adm.alpha_global - 1.0) < 1.0e-15);
    GDM_CHECK(res_adm.min_jacobian_after >= params.epsilon - 1.0e-12);

    // Spostamento severo: il filtro deve ridurre alpha e conservare i Jacobiani.
    const std::vector<MeshState::Coord2D> inverting_disp = {
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.8, 0.8}
    };
    const auto res_crit = apply_safety_filter(valid_nodes, cells, inverting_disp, boundary, params);
    GDM_CHECK(!res_crit.is_safe);
    GDM_CHECK(res_crit.filter_activated);
    GDM_CHECK(res_crit.alpha_global > 0.0 && res_crit.alpha_global < 1.0);
    GDM_CHECK(res_crit.min_jacobian_after >= params.epsilon - 1.0e-12);

    // Caso vicino alla soglia: un piccolo spostamento negativo attraverserebbe epsilon.
    const std::vector<MeshState::Coord2D> guard_nodes = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0e-4 + 0.5e-6}
    };
    const std::vector<MeshState::CellConn> one_cell = {{0, 1, 2}};
    const std::vector<MeshState::Coord2D> guard_disp = {
        {0.0, 0.0}, {0.0, 0.0}, {0.0, -1.0e-5}
    };
    const auto res_guard = apply_safety_filter(guard_nodes, one_cell, guard_disp, {0, 1}, params);
    GDM_CHECK(res_guard.filter_activated);
    GDM_CHECK(res_guard.alpha_global > 0.0 && res_guard.alpha_global < 1.0);
    GDM_CHECK(res_guard.min_jacobian_after >= params.epsilon - 1.0e-12);

    // Boundary lock, se esplicitamente abilitato, deve azzerare lo spostamento sul bordo.
    SafetyParameters locked = params;
    locked.lock_boundary_nodes = true;
    const auto res_locked = apply_safety_filter(valid_nodes, cells, small_disp, boundary, locked);
    GDM_CHECK(res_locked.safe_nodes[0] == valid_nodes[0]);
    GDM_CHECK(res_locked.safe_nodes[1] == valid_nodes[1]);
    GDM_CHECK(res_locked.safe_nodes[2] == valid_nodes[2]);

    return 0;
}
