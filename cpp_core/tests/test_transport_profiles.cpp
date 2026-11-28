#include "test_utils.hpp"
#include "gdm/pde_model.hpp"

#include <cmath>

int main() {
    using gdm::PDEParameters;
    PDEParameters params;
    params.bx = -0.35;
    params.by = 0.6;
    params.nu = 0.003;
    params.turn = 0.4;
    const double t = 0.2;
    const double x = 0.3;
    const double y = 0.48;
    const double h = 1e-5;
    for (const auto profile : {PDEParameters::Profile::Transport, PDEParameters::Profile::Turning}) {
        params.profile = profile;
        gdm::ManufacturedConvectionDiffusion2D pde(params);
        const auto ex = pde.evaluate_exact(x, y, t);
        const auto u = [&](double xx, double yy, double tt) { return pde.evaluate_solution(xx, yy, tt); };
        const double ut = (u(x, y, t + h) - u(x, y, t - h)) / (2 * h);
        const double ux = (u(x + h, y, t) - u(x - h, y, t)) / (2 * h);
        const double uy = (u(x, y + h, t) - u(x, y - h, t)) / (2 * h);
        const double lap = (u(x + h, y, t) + u(x - h, y, t) +
                            u(x, y + h, t) + u(x, y - h, t) - 4 * ex.u) / (h * h);
        GDM_CHECK(std::abs(ut - ex.ut) < 1e-7);
        GDM_CHECK(std::abs(ut + params.bx * ux + params.by * uy - params.nu * lap - ex.f) < 1e-6);
        if (profile == PDEParameters::Profile::Transport) {
            GDM_CHECK(ex.f == 0.0);
            const auto [cx, cy] = pde.get_center(t);
            GDM_CHECK(std::abs(cx - (params.center_x + params.bx * t)) < 1e-14);
            GDM_CHECK(std::abs(cy - (params.center_y + params.by * t)) < 1e-14);
        } else {
            GDM_CHECK(std::abs(ex.f) > 1e-4);
        }
    }
}
