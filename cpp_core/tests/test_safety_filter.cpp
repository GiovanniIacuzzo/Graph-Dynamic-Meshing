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

    // Numerical guard: with no alpha safety contraction, the path is still
    // constrained by epsilon + a scale-aware determinant buffer.
    SafetyParameters buffered = params;
    buffered.epsilon = 0.1;
    buffered.safety_margin = 0.0;
    buffered.relative_jacobian_margin = 1.0e-6;
    const std::vector<MeshState::Coord2D> buffer_nodes = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}
    };
    const std::vector<MeshState::CellConn> buffer_cells = {{0, 1, 2}};
    const std::vector<MeshState::Coord2D> buffer_disp = {
        {0.0, 0.0}, {0.0, 0.0}, {0.0, -2.0}
    };
    const auto res_buffered = apply_safety_filter(
        buffer_nodes, buffer_cells, buffer_disp, {0, 1}, buffered);
    GDM_CHECK(res_buffered.min_jacobian_after > buffered.epsilon);
    GDM_CHECK(res_buffered.min_jacobian_after >= buffered.epsilon + 0.5e-6);
    GDM_CHECK(res_buffered.alpha_global > 0.0 && res_buffered.alpha_global < 1.0);

    // If the initial element is admissible but has no room for the numerical
    // buffer, the filter must preserve the mesh (alpha=0), not relax epsilon.
    const std::vector<MeshState::Coord2D> near_floor_nodes = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.10000000005}
    };
    const std::vector<MeshState::Coord2D> near_floor_disp = {
        {0.0, 0.0}, {0.0, 0.0}, {0.0, -1.0e-3}
    };
    buffered.epsilon = 0.1;
    buffered.relative_jacobian_margin = 1.0e-6;
    const auto res_near_floor = apply_safety_filter(
        near_floor_nodes, buffer_cells, near_floor_disp, {0, 1}, buffered);
    GDM_CHECK(res_near_floor.alpha_global == 0.0);
    GDM_CHECK(res_near_floor.min_jacobian_after >= buffered.epsilon);

    const std::vector<MeshState::Coord2D> restoring_disp = {
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0e-3}
    };
    const auto restored = apply_safety_filter(
        near_floor_nodes, buffer_cells, restoring_disp, {0, 1}, buffered);
    GDM_CHECK(restored.alpha_global == 1.0);
    GDM_CHECK(restored.min_jacobian_after > buffered.epsilon + 1.0e-6);

    // An initially inadmissible mesh is rejected strictly, including a
    // sub-tolerance violation; the filter never treats it as a valid starting point.
    const std::vector<MeshState::Coord2D> below_floor_nodes = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0999999999999999}
    };
    const std::vector<MeshState::Coord2D> below_floor_disp(3, {0.0, 0.0});
    bool rejected_below_floor = false;
    try {
        (void)apply_safety_filter(
            below_floor_nodes, buffer_cells, below_floor_disp, {0, 1}, buffered);
    } catch (const std::invalid_argument&) {
        rejected_below_floor = true;
    }
    GDM_CHECK(rejected_below_floor);

    // Both endpoints are above epsilon, but the path dips below epsilon.
    // This also exercises coefficients close to the legacy root tolerance.
    const double floor = locked.epsilon;
    const double x = (floor + 1.3e-13) / 0.01 + 0.01;
    const double dx = -1.05e-6;
    const double dy = (-9.9e-13 + 0.01 * (1.0e-6 + dx)) / x;
    const std::vector<MeshState::Coord2D> path_nodes = {
        {0, 0}, {1, 0}, {1, 1}, {0, 1}, {x, 0.01}, {0.01, 0.01}
    };
    const std::vector<MeshState::CellConn> path_cells = {
        {2, 4, 1}, {1, 4, 0}, {3, 4, 2}, {5, 3, 0}, {4, 5, 0}, {5, 4, 3}
    };
    const std::vector<MeshState::Coord2D> path_displacement = {
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 1.0e-6}, {dx, dy}
    };
    const auto path = apply_safety_filter(path_nodes, path_cells, path_displacement, {0, 1, 2, 3}, locked);
    GDM_CHECK(path.filter_activated);
    GDM_CHECK(path.alpha_global > 0.0 && path.alpha_global < 0.47);
    GDM_CHECK(path.min_jacobian_after >= floor);

    return 0;
}
