#include "gdm/mesh_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gdm {

MeshQualityMetrics compute_mesh_metrics(
    const std::vector<MeshState::Coord2D>& nodes,
    const std::vector<MeshState::CellConn>& cells,
    double tol_degenerate) {
    if (nodes.size() < 3 || cells.empty()) {
        throw std::invalid_argument("compute_mesh_metrics: input vuoto o non valido.");
    }
    if (!std::isfinite(tol_degenerate) || tol_degenerate <= 0.0) {
        throw std::invalid_argument("compute_mesh_metrics: tol_degenerate deve essere > 0.");
    }

    const std::size_t m = cells.size();
    const auto n_nodes = static_cast<std::int64_t>(nodes.size());

    MeshQualityMetrics out;
    out.jacobians.resize(m);
    out.areas.resize(m);
    out.normalized_jacobians.resize(m);
    out.aspect_ratio.resize(m);
    out.min_angle.resize(m);
    out.max_angle.resize(m);
    out.escobar_quality.resize(m);
    out.inverted.resize(m, false);
    out.degenerate.resize(m, false);

    out.min_jacobian = std::numeric_limits<double>::infinity();
    out.max_jacobian = -std::numeric_limits<double>::infinity();
    out.min_escobar_quality = std::numeric_limits<double>::infinity();
    out.max_aspect_ratio = 0.0;

    const double sqrt3 = std::sqrt(3.0);
    double sum_escobar = 0.0;

    for (std::size_t e = 0; e < m; ++e) {
        const auto i1 = cells[e][0];
        const auto i2 = cells[e][1];
        const auto i3 = cells[e][2];
        if (i1 < 0 || i2 < 0 || i3 < 0 || i1 >= n_nodes || i2 >= n_nodes || i3 >= n_nodes) {
            throw std::out_of_range("compute_mesh_metrics: indice nodale fuori range.");
        }

        const double x1 = nodes[static_cast<std::size_t>(i1)][0];
        const double y1 = nodes[static_cast<std::size_t>(i1)][1];
        const double x2 = nodes[static_cast<std::size_t>(i2)][0];
        const double y2 = nodes[static_cast<std::size_t>(i2)][1];
        const double x3 = nodes[static_cast<std::size_t>(i3)][0];
        const double y3 = nodes[static_cast<std::size_t>(i3)][1];

        const double v12_x = x2 - x1;
        const double v12_y = y2 - y1;
        const double v23_x = x3 - x2;
        const double v23_y = y3 - y2;
        const double v31_x = x1 - x3;
        const double v31_y = y1 - y3;
        const double v13_x = x3 - x1;
        const double v13_y = y3 - y1;

        // Determinante Jacobiano orientato (senza valore assoluto)
        const double j_e = v12_x * v13_y - v13_x * v12_y;
        const double area_e = 0.5 * j_e;
        const bool is_inv = (j_e <= 0.0);

        const double l12_sq = v12_x * v12_x + v12_y * v12_y;
        const double l23_sq = v23_x * v23_x + v23_y * v23_y;
        const double l31_sq = v31_x * v31_x + v31_y * v31_y;

        const double l12 = std::sqrt(std::max(l12_sq, tol_degenerate));
        const double l23 = std::sqrt(std::max(l23_sq, tol_degenerate));
        const double l31 = std::sqrt(std::max(l31_sq, tol_degenerate));

        const bool is_deg = (area_e <= tol_degenerate) ||
                            (l12 <= tol_degenerate) ||
                            (l23 <= tol_degenerate) ||
                            (l31 <= tol_degenerate);

        const double sum_l_sq = std::max(l12_sq + l23_sq + l31_sq, tol_degenerate);
        const double eta_raw = (2.0 * sqrt3 * j_e) / sum_l_sq;
        const double eta = std::clamp(eta_raw, 0.0, 1.0);

        const double s = 0.5 * (l12 + l23 + l31);
        const double safe_area = std::max(area_e, tol_degenerate);
        const double r_in = safe_area / std::max(s, tol_degenerate);
        const double l_max = std::max({l12, l23, l31});
        const double ar = l_max / (2.0 * sqrt3 * std::max(r_in, tol_degenerate));

        const double j_ideal = (4.0 * s * s) / (6.0 * sqrt3);
        const double norm_j = j_e / std::max(j_ideal, tol_degenerate);

        const double c1 = std::clamp((v12_x * v13_x + v12_y * v13_y) / std::max(l12 * l31, tol_degenerate), -1.0, 1.0);
        const double c2 = std::clamp((-v12_x * v23_x - v12_y * v23_y) / std::max(l12 * l23, tol_degenerate), -1.0, 1.0);
        const double c3 = std::clamp((-v13_x * v23_x - v13_y * v23_y) / std::max(l31 * l23, tol_degenerate), -1.0, 1.0);

        const double th1 = std::acos(c1);
        const double th2 = std::acos(c2);
        const double th3 = std::acos(c3);

        out.jacobians[e] = j_e;
        out.areas[e] = area_e;
        out.normalized_jacobians[e] = norm_j;
        out.aspect_ratio[e] = ar;
        out.min_angle[e] = std::min({th1, th2, th3});
        out.max_angle[e] = std::max({th1, th2, th3});
        out.escobar_quality[e] = eta;
        out.inverted[e] = is_inv;
        out.degenerate[e] = is_deg;

        out.min_jacobian = std::min(out.min_jacobian, j_e);
        out.max_jacobian = std::max(out.max_jacobian, j_e);
        out.min_escobar_quality = std::min(out.min_escobar_quality, eta);
        out.max_aspect_ratio = std::max(out.max_aspect_ratio, ar);
        sum_escobar += eta;
        if (is_inv) ++out.num_inverted;
        if (is_deg) ++out.num_degenerate;
    }

    out.mean_escobar_quality = sum_escobar / static_cast<double>(m);
    return out;
}

MeshQualityMetrics compute_mesh_metrics(const MeshState& mesh, double tol_degenerate) {
    return compute_mesh_metrics(mesh.nodes(), mesh.cells(), tol_degenerate);
}

DisplacementComparisonMetrics compare_mesh_displacement(
    const std::vector<MeshState::Coord2D>& nodes_before,
    const std::vector<MeshState::Coord2D>& nodes_after,
    const std::vector<MeshState::CellConn>& cells,
    double tol_degenerate) {
    if (nodes_before.size() != nodes_after.size()) {
        throw std::invalid_argument("compare_mesh_displacement: numero di nodi differente.");
    }
    DisplacementComparisonMetrics comp;
    comp.before = compute_mesh_metrics(nodes_before, cells, tol_degenerate);
    comp.after = compute_mesh_metrics(nodes_after, cells, tol_degenerate);

    double sum_norm = 0.0;
    double max_norm = 0.0;
    for (std::size_t i = 0; i < nodes_before.size(); ++i) {
        const double dx = nodes_after[i][0] - nodes_before[i][0];
        const double dy = nodes_after[i][1] - nodes_before[i][1];
        const double d = std::hypot(dx, dy);
        sum_norm += d;
        max_norm = std::max(max_norm, d);
    }
    comp.max_displacement_norm = max_norm;
    comp.mean_displacement_norm = sum_norm / static_cast<double>(nodes_before.size());
    return comp;
}

} // namespace gdm