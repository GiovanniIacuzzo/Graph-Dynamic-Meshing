#pragma once

#include "gdm/mesh_state.hpp"

#include <cstddef>
#include <vector>

namespace gdm {

struct MeshQualityMetrics {
    std::vector<double> jacobians;
    std::vector<double> areas;
    std::vector<double> normalized_jacobians;
    std::vector<double> aspect_ratio;
    std::vector<double> min_angle;
    std::vector<double> max_angle;
    std::vector<double> escobar_quality;
    std::vector<bool> inverted;
    std::vector<bool> degenerate;

    double min_jacobian{0.0};
    double max_jacobian{0.0};
    double mean_escobar_quality{0.0};
    double min_escobar_quality{0.0};
    double max_aspect_ratio{0.0};
    std::size_t num_inverted{0};
    std::size_t num_degenerate{0};
};

struct DisplacementComparisonMetrics {
    MeshQualityMetrics before;
    MeshQualityMetrics after;
    double max_displacement_norm{0.0};
    double mean_displacement_norm{0.0};
};

MeshQualityMetrics compute_mesh_metrics(
    const std::vector<MeshState::Coord2D>& nodes,
    const std::vector<MeshState::CellConn>& cells,
    double tol_degenerate = 1.0e-14);

MeshQualityMetrics compute_mesh_metrics(
    const MeshState& mesh,
    double tol_degenerate = 1.0e-14);

DisplacementComparisonMetrics compare_mesh_displacement(
    const std::vector<MeshState::Coord2D>& nodes_before,
    const std::vector<MeshState::Coord2D>& nodes_after,
    const std::vector<MeshState::CellConn>& cells,
    double tol_degenerate = 1.0e-14);

} // namespace gdm