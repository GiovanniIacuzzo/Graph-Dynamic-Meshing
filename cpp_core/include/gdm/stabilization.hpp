#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace gdm {

// Residual-based streamline-upwind/Petrov-Galerkin (SUPG) options.
// The default is the historical, unstabilized Galerkin formulation.
struct StabilizationParameters {
    bool use_supg{false};
    double c_time{2.0};
    double c_advection{2.0};
    double c_diffusion{4.0};

    void validate() const {
        if (!std::isfinite(c_time) || !std::isfinite(c_advection) ||
            !std::isfinite(c_diffusion) || c_time <= 0.0 ||
            c_advection <= 0.0 || c_diffusion <= 0.0) {
            throw std::invalid_argument(
                "StabilizationParameters: i coefficient devono essere finiti e strettamente positivi.");
        }
    }
};

// Directional element length for affine P1 triangles:
// h_beta = 2 |beta| / sum_i | beta . grad(phi_i) |.
// For beta=0 the directional scale is undefined but SUPG's streamline
// test function is identically zero; return the element diameter as a
// harmless finite fallback for diagnostics/parameter evaluation.
inline double directional_element_length(
    const std::array<std::array<double, 2>, 3>& grad_phi,
    const std::array<double, 2>& beta,
    const std::array<std::array<double, 2>, 3>& vertices) {
    const double speed = std::hypot(beta[0], beta[1]);
    double max_edge = 0.0;
    for (unsigned int i = 0; i < 3; ++i) {
        const unsigned int j = (i + 1) % 3;
        max_edge = std::max(max_edge,
            std::hypot(vertices[j][0] - vertices[i][0],
                       vertices[j][1] - vertices[i][1]));
    }
    if (speed <= 1.0e-14) {
        return max_edge;
    }

    double denominator = 0.0;
    for (const auto& grad : grad_phi) {
        denominator += std::abs(beta[0] * grad[0] + beta[1] * grad[1]);
    }
    if (!std::isfinite(denominator) || denominator <= 1.0e-14 * speed) {
        throw std::runtime_error(
            "directional_element_length: denominatore nullo/non finito su triangolo non degenere.");
    }
    return 2.0 * speed / denominator;
}

inline double supg_tau(
    const double dt,
    const double speed,
    const double h_beta,
    const double nu,
    const StabilizationParameters& parameters) {
    if (!std::isfinite(dt) || !std::isfinite(speed) ||
        !std::isfinite(h_beta) || !std::isfinite(nu) ||
        dt <= 0.0 || speed < 0.0 || h_beta <= 0.0 || nu < 0.0) {
        throw std::invalid_argument("supg_tau: argomenti non validi.");
    }
    parameters.validate();
    const double temporal = parameters.c_time / dt;
    const double advective = parameters.c_advection * speed / h_beta;
    const double diffusive = parameters.c_diffusion * nu / (h_beta * h_beta);
    const double inverse_tau_squared = temporal * temporal +
                                       advective * advective +
                                       diffusive * diffusive;
    return 1.0 / std::sqrt(inverse_tau_squared);
}

} // namespace gdm
