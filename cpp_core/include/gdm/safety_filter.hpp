#pragma once

#include "gdm/mesh_state.hpp"
#include "gdm/parameters.hpp"

#include <cstddef>
#include <vector>

namespace gdm {

struct SafetyFilterResult {
    double alpha_global{1.0};
    std::vector<MeshState::Coord2D> safe_nodes;
    bool is_safe{true};
    double min_jacobian_before{0.0};
    double min_jacobian_after{0.0};
    std::size_t limiting_cell_index{0};
    bool filter_activated{false};
};

SafetyFilterResult apply_safety_filter(
    const std::vector<MeshState::Coord2D>& nodes,
    const std::vector<MeshState::CellConn>& cells,
    const std::vector<MeshState::Coord2D>& displacement,
    const std::vector<std::int64_t>& boundary_nodes,
    const SafetyParameters& params = SafetyParameters{});

SafetyFilterResult apply_safety_filter(
    const MeshState& mesh,
    const std::vector<MeshState::Coord2D>& displacement,
    const SafetyParameters& params = SafetyParameters{});

} // namespace gdm