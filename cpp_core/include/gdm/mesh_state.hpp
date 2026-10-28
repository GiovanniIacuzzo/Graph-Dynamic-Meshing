#pragma once

#include "gdm/parameters.hpp"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/tria.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace gdm {

class MeshState {
public:
    using Coord2D = std::array<double, 2>;
    using CellConn = std::array<std::int64_t, 3>;
    using EdgePair = std::array<std::int64_t, 2>;

    MeshState(const std::vector<Coord2D>& nodes,
              const std::vector<CellConn>& cells,
              const std::vector<std::int64_t>& boundary_nodes = {});

    static MeshState create_structured_mesh(const DomainParameters& domain);

    // Costruttori e assegnamenti di copia e move sicuri: ricostruiscono sempre DoFHandler su this->tria_
    MeshState(const MeshState& other);
    MeshState& operator=(const MeshState& other);
    MeshState(MeshState&& other);
    MeshState& operator=(MeshState&& other);

    std::size_t num_nodes() const noexcept { return nodes_.size(); }
    std::size_t num_cells() const noexcept { return cells_.size(); }

    const std::vector<Coord2D>& nodes() const noexcept { return nodes_; }
    const std::vector<CellConn>& cells() const noexcept { return cells_; }
    const std::vector<std::int64_t>& boundary_nodes() const noexcept { return boundary_nodes_; }
    const std::vector<bool>& is_boundary_node() const noexcept { return is_boundary_; }

    // Archi non orientati [E, 2], ordinati con u < v.
    const std::vector<EdgePair>& undirected_edges() const noexcept { return edges_; }
    // Archi orientati bidirezionali [2E, 2] in ordine deterministico per il grafo GNN.
    const std::vector<EdgePair>& directed_edges() const noexcept { return directed_edges_; }

    void update_coordinates(const std::vector<Coord2D>& new_nodes, bool enforce_fixed_boundary = false);
    void apply_displacement(const std::vector<Coord2D>& displacement, bool enforce_fixed_boundary = true);
    MeshState cloned_with_coordinates(const std::vector<Coord2D>& new_nodes,
                                      bool enforce_fixed_boundary = false) const;

    bool has_same_topology(const MeshState& other) const noexcept;
    void verify_positive_jacobians(double min_det_tol = 1.0e-15) const;

    const dealii::Triangulation<2>& triangulation() const noexcept { return tria_; }
    const dealii::DoFHandler<2>& dof_handler() const noexcept { return dof_handler_; }
    const dealii::FE_SimplexP<2>& fe() const noexcept { return fe_; }
    const dealii::MappingFE<2>& mapping() const noexcept { return mapping_; }

    std::size_t py_node_to_deal_vertex(std::size_t py_idx) const;
    std::size_t deal_vertex_to_py_node(std::size_t deal_v_idx) const;
    dealii::types::global_dof_index py_node_to_dof(std::size_t py_idx) const;
    std::size_t dof_to_py_node(dealii::types::global_dof_index dof_idx) const;

    std::uint64_t geometry_revision() const noexcept { return geometry_revision_; }

private:
    void validate_inputs() const;
    void detect_boundary_if_empty();
    void build_graph_edges();
    void build_dealii_structures();
    void sync_dealii_vertices();
    void verify_vertex_and_dof_mapping() const;

    std::vector<Coord2D> nodes_;
    std::vector<CellConn> cells_;
    std::vector<std::int64_t> boundary_nodes_;
    std::vector<bool> is_boundary_;
    std::vector<EdgePair> edges_;
    std::vector<EdgePair> directed_edges_;

    dealii::FE_SimplexP<2> fe_;
    dealii::MappingFE<2> mapping_;
    dealii::Triangulation<2> tria_;
    dealii::DoFHandler<2> dof_handler_;

    std::vector<std::size_t> py_to_deal_vertex_;
    std::vector<std::size_t> deal_to_py_vertex_;
    std::vector<dealii::types::global_dof_index> py_to_dof_;
    std::vector<std::size_t> dof_to_py_;
    std::uint64_t geometry_revision_{0};
};

} // namespace gdm