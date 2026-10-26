#include "test_utils.hpp"
#include "gdm/mesh_state.hpp"
#include "gdm/parameters.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
    using namespace gdm;

    DomainParameters dom;
    dom.nx = 8;
    dom.ny = 8;
    MeshState mesh = MeshState::create_structured_mesh(dom);

    GDM_CHECK(mesh.num_nodes() == 81);
    GDM_CHECK(mesh.num_cells() == 128);
    GDM_CHECK(mesh.boundary_nodes().size() == 32);
    GDM_CHECK(mesh.dof_handler().n_dofs() == 81);

    // Verifica biiezione tra nodi Python, vertici deal.II e DoF P1
    for (std::size_t py_i = 0; py_i < mesh.num_nodes(); ++py_i) {
        const std::size_t deal_v = mesh.py_node_to_deal_vertex(py_i);
        GDM_CHECK(mesh.deal_vertex_to_py_node(deal_v) == py_i);

        const auto dof = mesh.py_node_to_dof(py_i);
        GDM_CHECK(mesh.dof_to_py_node(dof) == py_i);
    }

    // Verifica blocco dei nodi di bordo durante apply_displacement
    const auto orig_nodes = mesh.nodes();
    std::vector<MeshState::Coord2D> disp(mesh.num_nodes(), {0.01, -0.01});
    const auto rev_before = mesh.geometry_revision();
    mesh.apply_displacement(disp, true);
    GDM_CHECK(mesh.geometry_revision() == rev_before + 1);

    const auto& moved_nodes = mesh.nodes();
    const auto& is_bnd = mesh.is_boundary_node();
    for (std::size_t i = 0; i < mesh.num_nodes(); ++i) {
        if (is_bnd[i]) {
            GDM_CHECK(std::abs(moved_nodes[i][0] - orig_nodes[i][0]) < 1.0e-15);
            GDM_CHECK(std::abs(moved_nodes[i][1] - orig_nodes[i][1]) < 1.0e-15);
        } else {
            GDM_CHECK(std::abs(moved_nodes[i][0] - (orig_nodes[i][0] + 0.01)) < 1.0e-15);
            GDM_CHECK(std::abs(moved_nodes[i][1] - (orig_nodes[i][1] - 0.01)) < 1.0e-15);
        }
    }

    // Verifica rifiuto di connettivita' fuori range
    bool threw_out_of_range = false;
    try {
        std::vector<MeshState::Coord2D> bad_nodes = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
        std::vector<MeshState::CellConn> bad_cells = {{0, 1, 5}};
        MeshState invalid_mesh(bad_nodes, bad_cells);
    } catch (const std::out_of_range&) {
        threw_out_of_range = true;
    }
    GDM_CHECK(threw_out_of_range);

    // Una maschera boundary fornita dal chiamante deve coincidere con il bordo topologico.
    bool threw_bad_boundary = false;
    try {
        const std::vector<MeshState::Coord2D> tri_nodes = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
        const std::vector<MeshState::CellConn> tri_cells = {{0, 1, 2}};
        MeshState invalid_boundary(tri_nodes, tri_cells, {0});
    } catch (const std::invalid_argument&) {
        threw_bad_boundary = true;
    }
    GDM_CHECK(threw_bad_boundary);

    // Un aggiornamento che invertirebbe elementi deve fallire senza mutare mesh o revision.
    const auto nodes_before_rejected_update = mesh.nodes();
    const auto revision_before_rejected_update = mesh.geometry_revision();
    auto invalid_coordinates = mesh.nodes();
    invalid_coordinates[40] = {4.0, 4.0};
    bool rejected_invalid_coordinates = false;
    try {
        mesh.update_coordinates(invalid_coordinates, false);
    } catch (const std::invalid_argument&) {
        rejected_invalid_coordinates = true;
    }
    GDM_CHECK(rejected_invalid_coordinates);
    GDM_CHECK(mesh.nodes() == nodes_before_rejected_update);
    GDM_CHECK(mesh.geometry_revision() == revision_before_rejected_update);

    return 0;
}