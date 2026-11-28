#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace gdm {

struct DomainParameters {
    double x_min{0.0};
    double x_max{1.0};
    double y_min{0.0};
    double y_max{1.0};
    std::size_t nx{32};
    std::size_t ny{32};

    void validate() const {
        if (!std::isfinite(x_min) || !std::isfinite(x_max) ||
            !std::isfinite(y_min) || !std::isfinite(y_max)) {
            throw std::invalid_argument("DomainParameters: limiti spaziali non finiti.");
        }
        if (x_max <= x_min || y_max <= y_min) {
            throw std::invalid_argument("DomainParameters: intervallo spaziale degenere o invertito.");
        }
        if (nx < 2 || ny < 2) {
            throw std::invalid_argument("DomainParameters: nx e ny devono essere >= 2.");
        }
    }
};

struct PDEParameters {
    double bx{0.4};
    double by{0.2};
    double nu{1.0e-3};
    double sigma{0.06};
    double beta{0.5};
    double T{0.5};
    enum class Profile { LegacyManufactured, Transport, Turning };
    Profile profile{Profile::LegacyManufactured};
    double center_x{0.3};
    double center_y{0.35};
    double turn{0.0};

    void validate() const {
        if (!std::isfinite(bx) || !std::isfinite(by) || !std::isfinite(nu) ||
            !std::isfinite(sigma) || !std::isfinite(beta) || !std::isfinite(T) ||
            !std::isfinite(center_x) || !std::isfinite(center_y) || !std::isfinite(turn)) {
            throw std::invalid_argument("PDEParameters: rilevato parametro non finito.");
        }
        if (nu < 0.0) {
            throw std::invalid_argument("PDEParameters: diffusivita' nu non puo' essere negativa.");
        }
        if (sigma <= 0.0) {
            throw std::invalid_argument("PDEParameters: sigma deve essere strettamente positivo.");
        }
        if (T <= 0.0) {
            throw std::invalid_argument("PDEParameters: orizzonte T deve essere strettamente positivo.");
        }
    }
};

struct SafetyParameters {
    double epsilon{1.0e-4};
    double tol_deg{1.0e-12};
    double safety_margin{1.0e-6};
    // Relative numerical buffer added to the determinant floor on each cell.
    // delta_e = max(relative_jacobian_margin, 64*eps_machine)
    //           * max(|J_e(0)|, epsilon).
    double relative_jacobian_margin{1.0e-9};
    bool lock_boundary_nodes{false};
    bool verbose{false};

    void validate() const {
        if (!std::isfinite(epsilon) || !std::isfinite(tol_deg) ||
            !std::isfinite(safety_margin) || !std::isfinite(relative_jacobian_margin)) {
            throw std::invalid_argument("SafetyParameters: parametri non finiti.");
        }
        if (epsilon <= 0.0) {
            throw std::invalid_argument("SafetyParameters: epsilon deve essere > 0.");
        }
        if (tol_deg <= 0.0) {
            throw std::invalid_argument("SafetyParameters: tol_deg deve essere > 0.");
        }
        if (safety_margin < 0.0 || safety_margin >= 0.5) {
            throw std::invalid_argument("SafetyParameters: safety_margin deve appartenere a [0.0, 0.5).");
        }
        if (relative_jacobian_margin < 0.0 || relative_jacobian_margin >= 0.1) {
            throw std::invalid_argument(
                "SafetyParameters: relative_jacobian_margin deve appartenere a [0.0, 0.1).");
        }
    }
};

struct LinearSolverParameters {
    double rel_tol{1.0e-12};
    double abs_tol{1.0e-14};
    std::size_t max_iterations{4000};
    std::size_t gmres_krylov_dim{50};
    double ssor_omega{1.0};
    bool use_direct_solver{false};
    // Preserve historical left-preconditioned GMRES unless explicitly selected.
    bool right_preconditioning{false};
    // Historical ALE used max(1, ||rhs||_2); Stage E sets this floor to zero
    // to use the same rhs-relative tolerance convention as the fixed solver.
    double ale_rhs_norm_floor{1.0};

    void validate() const {
        if (!std::isfinite(rel_tol) || !std::isfinite(abs_tol) ||
            !std::isfinite(ssor_omega) || !std::isfinite(ale_rhs_norm_floor)) {
            throw std::invalid_argument("LinearSolverParameters: tolleranze non finite.");
        }
        if (rel_tol <= 0.0 || abs_tol <= 0.0) {
            throw std::invalid_argument("LinearSolverParameters: tolleranze devono essere > 0.");
        }
        if (max_iterations == 0 || gmres_krylov_dim == 0) {
            throw std::invalid_argument("LinearSolverParameters: limiti iterazioni devono essere > 0.");
        }
        if (ssor_omega <= 0.0 || ssor_omega >= 2.0) {
            throw std::invalid_argument("LinearSolverParameters: ssor_omega deve appartenere a (0, 2).");
        }
        if (ale_rhs_norm_floor < 0.0) {
            throw std::invalid_argument("LinearSolverParameters: ale_rhs_norm_floor deve essere >= 0.");
        }
    }
};

struct SimulationParameters {
    DomainParameters domain{};
    PDEParameters pde{};
    SafetyParameters safety{};
    LinearSolverParameters linear_solver{};
    double dt{0.005};
    std::size_t adapt_interval{10};

    void validate() const {
        domain.validate();
        pde.validate();
        safety.validate();
        linear_solver.validate();
        if (!std::isfinite(dt) || dt <= 0.0) {
            throw std::invalid_argument("SimulationParameters: dt deve essere finito e > 0.");
        }
        if (adapt_interval == 0) {
            throw std::invalid_argument("SimulationParameters: adapt_interval deve essere >= 1.");
        }
        const double n_steps = std::round(pde.T / dt);
        if (std::abs(n_steps * dt - pde.T) > 1.0e-10 * pde.T + 1.0e-12) {
            throw std::invalid_argument("SimulationParameters: T non e' multiplo intero di dt.");
        }
    }
};

} // namespace gdm