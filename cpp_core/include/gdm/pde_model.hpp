#pragma once

#include "gdm/parameters.hpp"

#include <deal.II/base/quadrature.h>

#include <array>
#include <utility>
#include <vector>

namespace gdm {

struct ExactPointEvaluation {
    double u{0.0};
    double ux{0.0};
    double uy{0.0};
    double uxx{0.0};
    double uyy{0.0};
    double ut{0.0};
    double f{0.0};
};

class ManufacturedConvectionDiffusion2D {
public:
    explicit ManufacturedConvectionDiffusion2D(const PDEParameters& params = PDEParameters{});

    const PDEParameters& parameters() const noexcept { return params_; }
    double nu() const noexcept { return params_.nu; }
    std::array<double, 2> velocity() const noexcept { return {params_.bx, params_.by}; }
    std::pair<double, double> get_center(double t) const noexcept;

    ExactPointEvaluation evaluate_exact(double x, double y, double t) const noexcept;
    double evaluate_solution(double x, double y, double t) const noexcept;
    std::array<double, 2> evaluate_gradient(double x, double y, double t) const noexcept;
    double evaluate_forcing(double x, double y, double t) const noexcept;

    std::vector<double> evaluate_solution_on_nodes(
        const std::vector<std::array<double, 2>>& nodes,
        double t) const;

private:
    PDEParameters params_;
};

// Regola di quadratura simmetrica di Dunavant di grado 5 (7 punti) sul triangolo di riferimento
dealii::Quadrature<2> create_dunavant_degree5_quadrature();

} // namespace gdm