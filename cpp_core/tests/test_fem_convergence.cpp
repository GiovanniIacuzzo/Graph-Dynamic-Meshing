#include "test_utils.hpp"
#include "gdm/mms.hpp"

#include <vector>

int main() {
    using namespace gdm;

    // Raffinamento spaziale su mesh 8x8, 16x16, 32x32, 64x64 con dt ~ h^2
    const std::vector<std::size_t> grids = {8, 16, 32, 64};
    const auto rows = run_fem_spatial_convergence_study(grids, PDEParameters{}, 0.05, 1.0e-3);

    GDM_CHECK(rows.size() == 4);
    for (std::size_t i = 1; i < rows.size(); ++i) {
        GDM_CHECK(rows[i].rel_l2_error < rows[i - 1].rel_l2_error);
        GDM_CHECK(rows[i].rel_h1_error < rows[i - 1].rel_h1_error);
    }

    // Ordine asintotico atteso per elementi P1: p(L2) ~ 2.0, p(H1) ~ 1.0
    GDM_CHECK(rows.back().order_l2 > 1.85);
    GDM_CHECK(rows.back().order_h1 > 0.90);

    return 0;
}