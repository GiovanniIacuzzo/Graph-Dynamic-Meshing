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
    const double tol_deg = params.tol_deg;
    const double roundoff_tol = std::max(1.0e-12, 1.0e-9 * std::abs(eps));
    std::vector<double> A(n_cells, 0.0), B(n_cells, 0.0), C(n_cells, 0.0),
        C_tilde(n_cells, 0.0), alpha_e(n_cells, 1.0);
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
        C_tilde[e] = std::max(0.0, C[e] - eps);
        min_c = std::min(min_c, C[e]);
    }

    if (min_c < eps - roundoff_tol) {
        std::ostringstream msg;
        msg.precision(16);
        msg << "Mesh iniziale non ammissibile per il safety filter: min(J_0)="
            << std::scientific << min_c << " < epsilon=" << eps
            << " (tolleranza roundoff=" << roundoff_tol << ").";
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

    // Casi sulla soglia J(0)=epsilon.
    for (std::size_t e = 0; e < n_cells; ++e) {
        const double a = A[e];
        const double b = B[e];
        const double c = C_tilde[e];
        double alpha = 1.0;
        if (c <= 0.0) {
            if (b < 0.0 || (std::abs(b) <= tol_deg && a < 0.0)) {
                alpha = 0.0;
            } else if (b > 0.0 && a < 0.0 && b < -a) {
                alpha = std::clamp(-b / a, 0.0, 1.0);
            }
        } else if (std::abs(a) < tol_deg) {
            if (b < 0.0 && c + b <= 0.0) {
                alpha = std::clamp(-c / b, 0.0, 1.0);
            }
        } else {
            const double delta = b * b - 4.0 * a * c;
            if (!std::isfinite(delta)) {
                throw std::invalid_argument(
                    "apply_safety_filter: overflow numerico nel discriminante del determinante.");
            }
            const double scale_delta = (b * b + std::abs(4.0 * a * c) + 1.0) * tol_deg;
            if (!std::isfinite(scale_delta)) {
                throw std::invalid_argument(
                    "apply_safety_filter: overflow numerico nella tolleranza del discriminante.");
            }
            if (delta < 0.0 && delta >= -scale_delta && a > 0.0 && b < 0.0) {
                const double vertex = -b / (2.0 * a);
                if (vertex > 0.0 && vertex <= 1.0) {
                    alpha = vertex;
                }
            } else if (delta >= 0.0) {
                const double sqrt_delta = std::sqrt(std::max(0.0, delta));
                const double sign_b = (b >= 0.0) ? 1.0 : -1.0;
                double q = -0.5 * (b + sign_b * sqrt_delta);
                if (std::abs(b) < tol_deg) {
                    q = -0.5 * sqrt_delta;
                }
                double r1 = 2.0;
                double r2 = 2.0;
                if (std::abs(q) > 0.0) {
                    r1 = q / a;
                    r2 = c / q;
                }
                const double r1_valid = (r1 > 0.0 && r1 <= 1.0) ? r1 : 2.0;
                const double r2_valid = (r2 > 0.0 && r2 <= 1.0) ? r2 : 2.0;
                const double first = std::min(r1_valid, r2_valid);
                if (first <= 1.0) {
                    alpha = first;
                }
            }
        }
        alpha_e[e] = alpha;
        if (alpha < alpha_e[result.limiting_cell_index] || e == 0) {
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
    if (min_j_after < eps - roundoff_tol) {
        is_safe = false;
        if (alpha_global == 1.0) {
            alpha_global = 1.0 - std::max(params.safety_margin, 1.0e-6);
            min_j_after = minimum_jacobian_for_alpha(nodes, cells, disp, alpha_global, safe_nodes);
        }
        while (min_j_after < eps - roundoff_tol && alpha_global > 1.0e-16) {
            alpha_global *= 0.5;
            min_j_after = minimum_jacobian_for_alpha(nodes, cells, disp, alpha_global, safe_nodes);
        }
        if (min_j_after < eps - roundoff_tol) {
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
