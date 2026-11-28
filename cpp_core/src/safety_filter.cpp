#include "gdm/safety_filter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gdm {
namespace {

using Coord = MeshState::Coord2D;
using Cell = MeshState::CellConn;

inline double triangle_jacobian(const Coord& p0, const Coord& p1, const Coord& p2) {
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) -
           (p2[0] - p0[0]) * (p1[1] - p0[1]);
}

double minimum_jacobian_for_alpha(const std::vector<Coord>& nodes,
                                  const std::vector<Cell>& cells,
                                  const std::vector<Coord>& displacement,
                                  double alpha,
                                  std::vector<Coord>& safe_nodes) {
    safe_nodes.resize(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        safe_nodes[i] = {nodes[i][0] + alpha * displacement[i][0],
                         nodes[i][1] + alpha * displacement[i][1]};
    }

    double min_j = std::numeric_limits<double>::infinity();
    for (const auto& point : safe_nodes) {
        if (!std::isfinite(point[0]) || !std::isfinite(point[1])) {
            return -std::numeric_limits<double>::infinity();
        }
    }
    for (const auto& cell : cells) {
        const auto i0 = static_cast<std::size_t>(cell[0]);
        const auto i1 = static_cast<std::size_t>(cell[1]);
        const auto i2 = static_cast<std::size_t>(cell[2]);
        const double jacobian = triangle_jacobian(safe_nodes[i0], safe_nodes[i1], safe_nodes[i2]);
        if (!std::isfinite(jacobian)) {
            return -std::numeric_limits<double>::infinity();
        }
        min_j = std::min(min_j, jacobian);
    }
    return min_j;
}

double minimum_buffered_slack(const std::vector<Coord>& safe_nodes,
                              const std::vector<Cell>& cells,
                              const std::vector<double>& initial_jacobians,
                              const std::vector<double>& cell_buffers,
                              double epsilon) {
    double min_slack = std::numeric_limits<double>::infinity();
    for (std::size_t e = 0; e < cells.size(); ++e) {
        const auto i0 = static_cast<std::size_t>(cells[e][0]);
        const auto i1 = static_cast<std::size_t>(cells[e][1]);
        const auto i2 = static_cast<std::size_t>(cells[e][2]);
        const double jacobian = triangle_jacobian(
            safe_nodes[i0], safe_nodes[i1], safe_nodes[i2]);
        // Near the floor, preserve at least the initial determinant. A
        // motion that increases it may restore the full numerical buffer.
        const double required_floor = std::min(
            initial_jacobians[e], epsilon + cell_buffers[e]);
        min_slack = std::min(min_slack, jacobian - required_floor);
    }
    return min_slack;
}

}  // namespace

SafetyFilterResult apply_safety_filter(
    const std::vector<MeshState::Coord2D>& nodes,
    const std::vector<MeshState::CellConn>& cells,
    const std::vector<MeshState::Coord2D>& displacement,
    const std::vector<std::int64_t>& boundary_nodes,
    const SafetyParameters& params) {
    params.validate();
    if (nodes.size() < 3 || cells.empty() || displacement.size() != nodes.size()) {
        throw std::invalid_argument(
            "apply_safety_filter: dimensioni incoerenti di nodes/cells/displacement.");
    }

    const std::size_t n_nodes = nodes.size();
    const std::size_t n_cells = cells.size();
    for (std::size_t i = 0; i < n_nodes; ++i) {
        if (!std::isfinite(nodes[i][0]) || !std::isfinite(nodes[i][1]) ||
            !std::isfinite(displacement[i][0]) || !std::isfinite(displacement[i][1])) {
            throw std::invalid_argument(
                "apply_safety_filter: coordinate e spostamenti devono essere finiti.");
        }
    }
    for (const auto& cell : cells) {
        for (std::int64_t node : cell) {
            if (node < 0 || static_cast<std::size_t>(node) >= n_nodes) {
                throw std::out_of_range("apply_safety_filter: indice di cella fuori range.");
            }
        }
        if (cell[0] == cell[1] || cell[1] == cell[2] || cell[0] == cell[2]) {
            throw std::invalid_argument("apply_safety_filter: cella con nodi duplicati.");
        }
    }

    std::vector<Coord> disp = displacement;
    for (const std::int64_t b : boundary_nodes) {
        if (b < 0 || static_cast<std::size_t>(b) >= n_nodes) {
            throw std::out_of_range("apply_safety_filter: indice boundary_node fuori range.");
        }
    }
    if (params.lock_boundary_nodes) {
        for (const std::int64_t b : boundary_nodes) {
            disp[static_cast<std::size_t>(b)] = {0.0, 0.0};
        }
    }

    const double eps = params.epsilon;
    const double machine_guard = 64.0 * std::numeric_limits<double>::epsilon();
    const double relative_guard = std::max(params.relative_jacobian_margin, machine_guard);
    std::vector<double> A(n_cells, 0.0), B(n_cells, 0.0), C(n_cells, 0.0),
        C_buffered(n_cells, 0.0), delta_e(n_cells, 0.0), alpha_e(n_cells, 1.0);
    double min_c = std::numeric_limits<double>::infinity();
    double max_abs_disp = 0.0;

    for (const auto& d : disp) {
        max_abs_disp = std::max(max_abs_disp, std::max(std::abs(d[0]), std::abs(d[1])));
    }

    // Determinante lungo X(alpha)=X+alpha*d: J_e(alpha)=A_e alpha^2+B_e alpha+C_e.
    for (std::size_t e = 0; e < n_cells; ++e) {
        const auto i0 = static_cast<std::size_t>(cells[e][0]);
        const auto i1 = static_cast<std::size_t>(cells[e][1]);
        const auto i2 = static_cast<std::size_t>(cells[e][2]);
        const auto& p0 = nodes[i0];
        const auto& p1 = nodes[i1];
        const auto& p2 = nodes[i2];
        const auto& d0 = disp[i0];
        const auto& d1 = disp[i1];
        const auto& d2 = disp[i2];

        const double u1 = p1[0] - p0[0];
        const double u2 = p1[1] - p0[1];
        const double v1 = p2[0] - p0[0];
        const double v2 = p2[1] - p0[1];
        const double du1 = d1[0] - d0[0];
        const double du2 = d1[1] - d0[1];
        const double dv1 = d2[0] - d0[0];
        const double dv2 = d2[1] - d0[1];

        C[e] = u1 * v2 - u2 * v1;
        B[e] = (u1 * dv2 + du1 * v2) - (u2 * dv1 + du2 * v1);
        A[e] = du1 * dv2 - du2 * dv1;
        if (!std::isfinite(A[e]) || !std::isfinite(B[e]) || !std::isfinite(C[e])) {
            throw std::invalid_argument(
                "apply_safety_filter: overflow numerico nei coefficienti del determinante.");
        }
        const double local_scale = std::max(std::abs(C[e]), std::abs(eps));
        delta_e[e] = relative_guard * local_scale;
        const double buffered_floor = eps + delta_e[e];
        C_buffered[e] = C[e] - std::min(C[e], buffered_floor);
        min_c = std::min(min_c, C[e]);
    }

    // Do not silently accept an initially inadmissible mesh, even within a
    // roundoff tolerance. The buffer only restricts the proposed motion; it
    // does not relax the physical admissibility threshold epsilon.
    if (min_c < eps) {
        std::ostringstream msg;
        msg.precision(16);
        msg << "Mesh iniziale non ammissibile per il safety filter: min(J_0)="
            << std::scientific << min_c << " < epsilon=" << eps << ".";
        throw std::invalid_argument(msg.str());
    }

    SafetyFilterResult result{};
    result.min_jacobian_before = min_c;
    result.limiting_cell_index = 0;

    if (max_abs_disp == 0.0) {
        result.alpha_global = 1.0;
        result.safe_nodes = nodes;
        result.is_safe = true;
        result.min_jacobian_after = min_c;
        result.filter_activated = false;
        return result;
    }

    // The minimum over a prefix is monotone, even when both endpoints
    // are admissible and the quadratic dips below the floor in between.
    const auto admissible_prefix = [&](std::size_t e, double alpha) {
        const long double a = A[e];
        const long double b = B[e];
        const long double c = C_buffered[e];
        const auto value = [&](long double t) {
            return std::fma(std::fma(a, t, b), t, c);
        };
        if (value(alpha) < 0.0L) {
            return false;
        }
        if (a > 0.0L && b < 0.0L) {
            const long double vertex = -b / (2.0L * a);
            return vertex >= alpha || value(vertex) >= 0.0L;
        }
        return true;
    };
    for (std::size_t e = 0; e < n_cells; ++e) {
        if (C_buffered[e] == 0.0 && (B[e] < 0.0 || (B[e] == 0.0 && A[e] < 0.0))) {
            alpha_e[e] = 0.0;
        } else if (!admissible_prefix(e, 1.0)) {
            double low = 0.0;
            double high = 1.0;
            for (int iteration = 0; iteration < 64; ++iteration) {
                const double trial = 0.5 * (low + high);
                if (trial == low || trial == high) {
                    break;
                }
                if (admissible_prefix(e, trial)) {
                    low = trial;
                } else {
                    high = trial;
                }
            }
            alpha_e[e] = low;
        }
        if (alpha_e[e] < alpha_e[result.limiting_cell_index]) {
            result.limiting_cell_index = e;
        }
    }

    const auto alpha_it = std::min_element(alpha_e.begin(), alpha_e.end());
    const double alpha_raw = *alpha_it;
    double alpha_global = 1.0;
    bool is_safe = true;
    if (alpha_raw < 1.0) {
        alpha_global = std::max(0.0, (1.0 - std::clamp(params.safety_margin, 0.0, 0.5)) * alpha_raw);
        is_safe = false;
    }

    std::vector<Coord> safe_nodes;
    double min_j_after = minimum_jacobian_for_alpha(nodes, cells, disp, alpha_global, safe_nodes);
    double min_buffered_slack = minimum_buffered_slack(
        safe_nodes, cells, C, delta_e, eps);
    if (min_j_after < eps || min_buffered_slack < 0.0) {
        is_safe = false;
        if (alpha_global == 1.0) {
            alpha_global = 1.0 - std::max(params.safety_margin, 1.0e-6);
            min_j_after = minimum_jacobian_for_alpha(nodes, cells, disp, alpha_global, safe_nodes);
            min_buffered_slack = minimum_buffered_slack(
                safe_nodes, cells, C, delta_e, eps);
        }
        while ((min_j_after < eps || min_buffered_slack < 0.0) && alpha_global > 1.0e-16) {
            alpha_global *= 0.5;
            min_j_after = minimum_jacobian_for_alpha(nodes, cells, disp, alpha_global, safe_nodes);
            min_buffered_slack = minimum_buffered_slack(
                safe_nodes, cells, C, delta_e, eps);
        }
        if (min_j_after < eps || min_buffered_slack < 0.0) {
            alpha_global = 0.0;
            safe_nodes = nodes;
            min_j_after = min_c;
        }
    }

    result.alpha_global = alpha_global;
    result.safe_nodes = std::move(safe_nodes);
    result.is_safe = is_safe;
    result.min_jacobian_after = min_j_after;
    result.filter_activated = (alpha_global < 1.0 - 1.0e-12);
    return result;
}

SafetyFilterResult apply_safety_filter(
    const MeshState& mesh,
    const std::vector<MeshState::Coord2D>& displacement,
    const SafetyParameters& params) {
    return apply_safety_filter(
        mesh.nodes(), mesh.cells(), displacement, mesh.boundary_nodes(), params);
}

}  // namespace gdm
