#include "gdm/pde_model.hpp"

#include <deal.II/base/point.h>

#include <cmath>
#include <stdexcept>

namespace gdm {
namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
}

ManufacturedConvectionDiffusion2D::ManufacturedConvectionDiffusion2D(
    const PDEParameters& params)
    : params_(params) {
    params_.validate();
}

std::pair<double, double> ManufacturedConvectionDiffusion2D::get_center(
    double t) const noexcept {
    const double tau = t / params_.T;
    return {0.3 + 0.4 * tau, 0.35 + 0.3 * tau};
}

ExactPointEvaluation ManufacturedConvectionDiffusion2D::evaluate_exact(
    double x, double y, double t) const noexcept {
    const auto [cx, cy] = get_center(t);
    const double rx = x - cx;
    const double ry = y - cy;
    const double inv_sigma2 = 1.0 / (params_.sigma * params_.sigma);

    const double gaussian = std::exp(-0.5 * (rx * rx + ry * ry) * inv_sigma2);
    const double decay = std::exp(-params_.beta * t);

    const double sin_x = std::sin(kPi * x);
    const double cos_x = std::cos(kPi * x);
    const double sin_y = std::sin(kPi * y);
    const double cos_y = std::cos(kPi * y);
    const double shape = sin_x * sin_y;
    const double shape_x = kPi * cos_x * sin_y;
    const double shape_y = kPi * sin_x * cos_y;
    const double shape_xx = -kPi * kPi * shape;
    const double shape_yy = -kPi * kPi * shape;

    const double gaussian_x = -rx * inv_sigma2 * gaussian;
    const double gaussian_y = -ry * inv_sigma2 * gaussian;
    const double gaussian_xx = (rx * rx * inv_sigma2 * inv_sigma2 - inv_sigma2) * gaussian;
    const double gaussian_yy = (ry * ry * inv_sigma2 * inv_sigma2 - inv_sigma2) * gaussian;

    const double center_x_t = 0.4 / params_.T;
    const double center_y_t = 0.3 / params_.T;
    const double gaussian_t = (rx * center_x_t + ry * center_y_t) * inv_sigma2 * gaussian;

    ExactPointEvaluation out{};
    out.u = gaussian * shape * decay;
    out.ux = (gaussian_x * shape + gaussian * shape_x) * decay;
    out.uy = (gaussian_y * shape + gaussian * shape_y) * decay;
    out.uxx = (gaussian_xx * shape + 2.0 * gaussian_x * shape_x +
               gaussian * shape_xx) * decay;
    out.uyy = (gaussian_yy * shape + 2.0 * gaussian_y * shape_y +
               gaussian * shape_yy) * decay;
    out.ut = (gaussian_t - params_.beta * gaussian) * shape * decay;
    out.f = out.ut + params_.bx * out.ux + params_.by * out.uy -
            params_.nu * (out.uxx + out.uyy);
    return out;
}

double ManufacturedConvectionDiffusion2D::evaluate_solution(
    double x, double y, double t) const noexcept {
    return evaluate_exact(x, y, t).u;
}

std::array<double, 2> ManufacturedConvectionDiffusion2D::evaluate_gradient(
    double x, double y, double t) const noexcept {
    const auto eval = evaluate_exact(x, y, t);
    return {eval.ux, eval.uy};
}

double ManufacturedConvectionDiffusion2D::evaluate_forcing(
    double x, double y, double t) const noexcept {
    return evaluate_exact(x, y, t).f;
}

std::vector<double> ManufacturedConvectionDiffusion2D::evaluate_solution_on_nodes(
    const std::vector<std::array<double, 2>>& nodes, double t) const {
    std::vector<double> values(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        values[i] = evaluate_solution(nodes[i][0], nodes[i][1], t);
    }
    return values;
}

dealii::Quadrature<2> create_dunavant_degree5_quadrature() {
    // Dunavant grado 5; i pesi sono scalati a sommare 1/2, area del triangolo di riferimento.
    const double a1 = 0.059715871789770;
    const double b1 = 0.470142064105115;
    const double a2 = 0.797426985353087;
    const double b2 = 0.101286507323456;

    const double w0 = 0.225000000000000 * 0.5;
    const double w1 = 0.132394152788506 * 0.5;
    const double w2 = 0.125939180544827 * 0.5;

    std::vector<dealii::Point<2>> points = {
        dealii::Point<2>(1.0 / 3.0, 1.0 / 3.0),
        dealii::Point<2>(a1, b1), dealii::Point<2>(b1, a1), dealii::Point<2>(b1, b1),
        dealii::Point<2>(a2, b2), dealii::Point<2>(b2, a2), dealii::Point<2>(b2, b2)};
    std::vector<double> weights = {w0, w1, w1, w1, w2, w2, w2};
    return dealii::Quadrature<2>(points, weights);
}

}  // namespace gdm
