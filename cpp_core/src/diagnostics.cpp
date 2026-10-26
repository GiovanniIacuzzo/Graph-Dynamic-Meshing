#include "gdm/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gdm {

ContinuousErrorNorms compute_continuous_errors(
    const MeshState& mesh,
    const std::vector<double>& u_h,
    const ManufacturedConvectionDiffusion2D& pde,
    double t) {
    const auto& nodes = mesh.nodes();
    const auto& cells = mesh.cells();
    if (u_h.size() != nodes.size()) {
        throw std::invalid_argument("compute_continuous_errors: dimensione u_h incompatibile.");
    }

    const auto quad = create_dunavant_degree5_quadrature();
    const auto& q_pts = quad.get_points();
    const auto& q_wts = quad.get_weights();
    const std::size_t n_q = q_pts.size();

    double l2_err_sq = 0.0;
    double h1_err_sq = 0.0;
    double l2_ref_sq = 0.0;
    double h1_ref_sq = 0.0;

    for (const auto& c : cells) {
        const auto i1 = static_cast<std::size_t>(c[0]);
        const auto i2 = static_cast<std::size_t>(c[1]);
        const auto i3 = static_cast<std::size_t>(c[2]);

        const double x1 = nodes[i1][0], y1 = nodes[i1][1];
        const double x2 = nodes[i2][0], y2 = nodes[i2][1];
        const double x3 = nodes[i3][0], y3 = nodes[i3][1];

        const double det_j = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
        const double area = 0.5 * std::abs(det_j);
        const double inv_det = 1.0 / det_j;

        const double dphi1_dx = (y2 - y3) * inv_det;
        const double dphi2_dx = (y3 - y1) * inv_det;
        const double dphi3_dx = (y1 - y2) * inv_det;

        const double dphi1_dy = (x3 - x2) * inv_det;
        const double dphi2_dy = (x1 - x3) * inv_det;
        const double dphi3_dy = (x2 - x1) * inv_det;

        const double duh_dx = u_h[i1] * dphi1_dx + u_h[i2] * dphi2_dx + u_h[i3] * dphi3_dx;
        const double duh_dy = u_h[i1] * dphi1_dy + u_h[i2] * dphi2_dy + u_h[i3] * dphi3_dy;

        for (std::size_t q = 0; q < n_q; ++q) {
            const double xi = q_pts[q][0];
            const double eta = q_pts[q][1];
            const double phi1 = 1.0 - xi - eta;
            const double phi2 = xi;
            const double phi3 = eta;

            const double xq = phi1 * x1 + phi2 * x2 + phi3 * x3;
            const double yq = phi1 * y1 + phi2 * y2 + phi3 * y3;

            const auto exact = pde.evaluate_exact(xq, yq, t);
            const double u_fem_q = phi1 * u_h[i1] + phi2 * u_h[i2] + phi3 * u_h[i3];

            const double diff_u = u_fem_q - exact.u;
            const double diff_gx = duh_dx - exact.ux;
            const double diff_gy = duh_dy - exact.uy;

            const double jxw = (2.0 * q_wts[q]) * area;
            l2_err_sq += jxw * (diff_u * diff_u);
            h1_err_sq += jxw * (diff_gx * diff_gx + diff_gy * diff_gy);
            l2_ref_sq += jxw * (exact.u * exact.u);
            h1_ref_sq += jxw * (exact.ux * exact.ux + exact.uy * exact.uy);
        }
    }

    ContinuousErrorNorms out;
    out.abs_l2_error = std::sqrt(std::max(l2_err_sq, 0.0));
    out.abs_h1_seminorm_error = std::sqrt(std::max(h1_err_sq, 0.0));
    out.rel_l2_error = out.abs_l2_error / std::sqrt(std::max(l2_ref_sq, 1.0e-14));
    out.rel_h1_seminorm_error = out.abs_h1_seminorm_error / std::sqrt(std::max(h1_ref_sq, 1.0e-14));
    return out;
}

double compute_mass_matrix_l2_difference(
    const MeshState& mesh,
    const std::vector<double>& u_a,
    const std::vector<double>& u_b) {
    const auto& nodes = mesh.nodes();
    const auto& cells = mesh.cells();
    if (u_a.size() != nodes.size() || u_b.size() != nodes.size()) {
        throw std::invalid_argument("compute_mass_matrix_l2_difference: dimensioni vettori non valide.");
    }
    double integral_sq = 0.0;
    for (const auto& c : cells) {
        const auto i1 = static_cast<std::size_t>(c[0]);
        const auto i2 = static_cast<std::size_t>(c[1]);
        const auto i3 = static_cast<std::size_t>(c[2]);

        const double det_j = (nodes[i2][0] - nodes[i1][0]) * (nodes[i3][1] - nodes[i1][1]) -
                             (nodes[i3][0] - nodes[i1][0]) * (nodes[i2][1] - nodes[i1][1]);
        const double area = 0.5 * std::abs(det_j);

        const double d1 = u_a[i1] - u_b[i1];
        const double d2 = u_a[i2] - u_b[i2];
        const double d3 = u_a[i3] - u_b[i3];

        // Formula esatta per elementi P1 tramite la matrice di massa locale (area / 12)
        const double quad_form = 2.0 * (d1 * d1 + d2 * d2 + d3 * d3) +
                                 2.0 * (d1 * d2 + d2 * d3 + d3 * d1);
        integral_sq += (area / 12.0) * quad_form;
    }
    return std::sqrt(std::max(integral_sq, 0.0));
}

double compute_time_integrated_error(
    const std::vector<double>& times,
    const std::vector<double>& errors) {
    if (times.size() != errors.size() || times.size() < 2) {
        throw std::invalid_argument("compute_time_integrated_error: serie temporali non valide.");
    }
    const double total_t = times.back() - times.front();
    if (total_t <= 0.0) {
        throw std::invalid_argument("compute_time_integrated_error: intervallo temporale non positivo.");
    }
    double accum = 0.0;
    for (std::size_t i = 0; i + 1 < times.size(); ++i) {
        const double dt = times[i + 1] - times[i];
        accum += 0.5 * dt * (errors[i] + errors[i + 1]);
    }
    return accum / total_t;
}

double compute_free_stream_error_linf(const std::vector<double>& u_h, double reference_constant) {
    double max_err = 0.0;
    for (double v : u_h) {
        max_err = std::max(max_err, std::abs(v - reference_constant));
    }
    return max_err;
}

double compute_relative_mass_drift(double initial_mass, double current_mass) {
    return std::abs(current_mass - initial_mass) / std::max(std::abs(initial_mass), 1.0e-14);
}

} // namespace gdm