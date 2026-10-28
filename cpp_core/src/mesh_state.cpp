#include "gdm/mesh_state.hpp"

#include <deal.II/base/point.h>
#include <deal.II/grid/grid_tools.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace gdm {

MeshState::MeshState(const std::vector<Coord2D>& nodes,
                     const std::vector<CellConn>& cells,
                     const std::vector<std::int64_t>& boundary_nodes)
    : nodes_(nodes),
      cells_(cells),
      boundary_nodes_(boundary_nodes),
      fe_(1),
      mapping_(dealii::FE_SimplexP<2>(1)),
      dof_handler_(tria_) {
    validate_inputs();
    detect_boundary_if_empty();
    build_graph_edges();
    build_dealii_structures();
}

MeshState::MeshState(const MeshState& other)
    : nodes_(other.nodes_),
      cells_(other.cells_),
      boundary_nodes_(other.boundary_nodes_),
      is_boundary_(other.is_boundary_),
      edges_(other.edges_),
      directed_edges_(other.directed_edges_),
      fe_(1),
      mapping_(dealii::FE_SimplexP<2>(1)),
      dof_handler_(tria_),
      geometry_revision_(other.geometry_revision_) {
    build_dealii_structures();
}

MeshState& MeshState::operator=(const MeshState& other) {
    if (this != &other) {
        nodes_ = other.nodes_;
        cells_ = other.cells_;
        boundary_nodes_ = other.boundary_nodes_;
        is_boundary_ = other.is_boundary_;
        edges_ = other.edges_;
        directed_edges_ = other.directed_edges_;
        geometry_revision_ = other.geometry_revision_;
        build_dealii_structures();
    }
    return *this;
}

MeshState::MeshState(MeshState&& other)
    : nodes_(std::move(other.nodes_)),
      cells_(std::move(other.cells_)),
      boundary_nodes_(std::move(other.boundary_nodes_)),
      is_boundary_(std::move(other.is_boundary_)),
      edges_(std::move(other.edges_)),
      directed_edges_(std::move(other.directed_edges_)),
      fe_(1),
      mapping_(dealii::FE_SimplexP<2>(1)),
      dof_handler_(tria_),
      geometry_revision_(other.geometry_revision_) {
    build_dealii_structures();
}

MeshState& MeshState::operator=(MeshState&& other) {
    if (this != &other) {
        nodes_ = std::move(other.nodes_);
        cells_ = std::move(other.cells_);
        boundary_nodes_ = std::move(other.boundary_nodes_);
        is_boundary_ = std::move(other.is_boundary_);
        edges_ = std::move(other.edges_);
        directed_edges_ = std::move(other.directed_edges_);
        geometry_revision_ = other.geometry_revision_;
        build_dealii_structures();
    }
    return *this;
}

MeshState MeshState::create_structured_mesh(const DomainParameters& domain) {
    domain.validate();
    const std::size_t nx = domain.nx;
    const std::size_t ny = domain.ny;
    const std::size_t num_nodes = (nx + 1) * (ny + 1);

    std::vector<Coord2D> nodes(num_nodes);
    std::vector<std::int64_t> bnd;
    const double dx = (domain.x_max - domain.x_min) / static_cast<double>(nx);
    const double dy = (domain.y_max - domain.y_min) / static_cast<double>(ny);

    for (std::size_t j = 0; j <= ny; ++j) {
        for (std::size_t i = 0; i <= nx; ++i) {
            const std::size_t idx = j * (nx + 1) + i;
            nodes[idx] = {domain.x_min + static_cast<double>(i) * dx,
                          domain.y_min + static_cast<double>(j) * dy};
            if (i == 0 || i == nx || j == 0 || j == ny) {
                bnd.push_back(static_cast<std::int64_t>(idx));
            }
        }
    }

    std::vector<CellConn> cells;
    cells.reserve(2 * nx * ny);
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            const auto n00 = static_cast<std::int64_t>(j * (nx + 1) + i);
            const auto n10 = static_cast<std::int64_t>(j * (nx + 1) + (i + 1));
            const auto n01 = static_cast<std::int64_t>((j + 1) * (nx + 1) + i);
            const auto n11 = static_cast<std::int64_t>((j + 1) * (nx + 1) + (i + 1));
            cells.push_back({n00, n10, n11});
            cells.push_back({n00, n11, n01});
        }
    }

    return MeshState(nodes, cells, bnd);
}

void MeshState::validate_inputs() const {
    if (nodes_.size() < 3) {
        throw std::invalid_argument("MeshState: numero di nodi inferiore a 3.");
    }
    if (cells_.empty()) {
        throw std::invalid_argument("MeshState: connettivita' triangolare vuota.");
    }
    for (const auto& pt : nodes_) {
        if (!std::isfinite(pt[0]) || !std::isfinite(pt[1])) {
            throw std::invalid_argument("MeshState: rilevata coordinata nodale non finita.");
        }
    }
    const auto n_max = static_cast<std::int64_t>(nodes_.size());
    std::vector<bool> referenced(nodes_.size(), false);
    for (const auto& c : cells_) {
        if (c[0] < 0 || c[1] < 0 || c[2] < 0 ||
            c[0] >= n_max || c[1] >= n_max || c[2] >= n_max) {
            throw std::out_of_range("MeshState: indice di cella fuori range.");
        }
        if (c[0] == c[1] || c[1] == c[2] || c[2] == c[0]) {
            throw std::invalid_argument("MeshState: triangolo con vertici duplicati.");
        }
        referenced[static_cast<std::size_t>(c[0])] = true;
        referenced[static_cast<std::size_t>(c[1])] = true;
        referenced[static_cast<std::size_t>(c[2])] = true;
    }
    for (bool used : referenced) {
        if (!used) {
            throw std::invalid_argument("MeshState: presenza di nodi isolati non referenziati.");
        }
    }
    for (std::int64_t b : boundary_nodes_) {
        if (b < 0 || b >= n_max) {
            throw std::out_of_range("MeshState: nodo di bordo fuori range.");
        }
    }
}

void MeshState::verify_positive_jacobians(double min_det_tol) const {
    for (std::size_t e = 0; e < cells_.size(); ++e) {
        const auto i1 = static_cast<std::size_t>(cells_[e][0]);
        const auto i2 = static_cast<std::size_t>(cells_[e][1]);
        const auto i3 = static_cast<std::size_t>(cells_[e][2]);
        const double det_j =
            (nodes_[i2][0] - nodes_[i1][0]) * (nodes_[i3][1] - nodes_[i1][1]) -
            (nodes_[i3][0] - nodes_[i1][0]) * (nodes_[i2][1] - nodes_[i1][1]);
        if (!std::isfinite(det_j) || det_j <= min_det_tol) {
            throw std::runtime_error(
                "MeshState: rilevato elemento invertito o degenere (cell_index=" +
                std::to_string(e) + ", det_J=" + std::to_string(det_j) + ").");
        }
    }
}

void MeshState::detect_boundary_if_empty() {
    is_boundary_.assign(nodes_.size(), false);

    std::map<std::pair<std::int64_t, std::int64_t>, unsigned int> edge_counts;
    for (const auto& c : cells_) {
        const std::array<std::pair<std::int64_t, std::int64_t>, 3> cell_edges = {{
            {std::min(c[0], c[1]), std::max(c[0], c[1])},
            {std::min(c[1], c[2]), std::max(c[1], c[2])},
            {std::min(c[2], c[0]), std::max(c[2], c[0])}
        }};
        for (const auto& e : cell_edges) {
            ++edge_counts[e];
            if (edge_counts[e] > 2) {
                throw std::invalid_argument(
                    "MeshState: lato condiviso da piu' di due triangoli; mesh non conforme.");
            }
        }
    }

    std::set<std::int64_t> boundary_set;
    for (const auto& [edge, count] : edge_counts) {
        if (count == 1) {
            boundary_set.insert(edge.first);
            boundary_set.insert(edge.second);
        }
    }
    const std::vector<std::int64_t> derived_boundary(boundary_set.begin(), boundary_set.end());

    if (boundary_nodes_.empty()) {
        boundary_nodes_ = derived_boundary;
    } else {
        std::set<std::int64_t> supplied_set;
        for (const std::int64_t b : boundary_nodes_) {
            if (b < 0 || static_cast<std::size_t>(b) >= nodes_.size()) {
                throw std::out_of_range("MeshState: nodo di bordo fuori range.");
            }
            supplied_set.insert(b);
        }
        const std::vector<std::int64_t> supplied_boundary(supplied_set.begin(), supplied_set.end());
        if (supplied_boundary != derived_boundary) {
            throw std::invalid_argument(
                "MeshState: boundary_nodes non coincide con il bordo topologico ricavato dalle celle.");
        }
        boundary_nodes_ = supplied_boundary;
    }

    for (const std::int64_t b : boundary_nodes_) {
        is_boundary_[static_cast<std::size_t>(b)] = true;
    }
}

void MeshState::build_graph_edges() {
    std::set<std::pair<std::int64_t, std::int64_t>> edge_set;
    for (const auto& c : cells_) {
        edge_set.insert({std::min(c[0], c[1]), std::max(c[0], c[1])});
        edge_set.insert({std::min(c[1], c[2]), std::max(c[1], c[2])});
        edge_set.insert({std::min(c[2], c[0]), std::max(c[2], c[0])});
    }
    edges_.clear();
    edges_.reserve(edge_set.size());
    for (const auto& [u, v] : edge_set) {
        edges_.push_back({u, v});
    }

    // Costruzione di directed_edges [2E, 2] con ordinamento lessicografico,
    // con ordine deterministico, per garantire riproducibilita' del grafo.
    directed_edges_.clear();
    directed_edges_.reserve(2 * edges_.size());
    for (const auto& e : edges_) {
        directed_edges_.push_back({e[0], e[1]});
    }
    for (const auto& e : edges_) {
        directed_edges_.push_back({e[1], e[0]});
    }
    std::sort(directed_edges_.begin(), directed_edges_.end(),
              [](const EdgePair& a, const EdgePair& b) {
                  return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]);
              });
}

void MeshState::build_dealii_structures() {
    dof_handler_.clear();
    tria_.clear();

    std::vector<dealii::Point<2>> vertices(nodes_.size());
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        vertices[i] = dealii::Point<2>(nodes_[i][0], nodes_[i][1]);
    }

    std::vector<dealii::CellData<2>> deal_cells(cells_.size(), dealii::CellData<2>(3));
    for (std::size_t e = 0; e < cells_.size(); ++e) {
        deal_cells[e].vertices[0] = static_cast<unsigned int>(cells_[e][0]);
        deal_cells[e].vertices[1] = static_cast<unsigned int>(cells_[e][1]);
        deal_cells[e].vertices[2] = static_cast<unsigned int>(cells_[e][2]);
        deal_cells[e].material_id = 0;
    }

    tria_.create_triangulation(vertices, deal_cells, dealii::SubCellData());
    dof_handler_.distribute_dofs(fe_);

    const std::size_t n_nodes = nodes_.size();
    if (tria_.n_used_vertices() != n_nodes || dof_handler_.n_dofs() != n_nodes) {
        throw std::runtime_error("MeshState: disallineamento tra numero di nodi Python e DoF P1 deal.II.");
    }

    py_to_deal_vertex_.assign(n_nodes, std::numeric_limits<std::size_t>::max());
    deal_to_py_vertex_.assign(tria_.n_vertices(), std::numeric_limits<std::size_t>::max());

    // Associazione basata sulle celle + verifica di consistenza univoca tra celle adiacenti
    std::size_t cell_idx = 0;
    for (const auto& cell : tria_.active_cell_iterators()) {
        if (cell_idx >= cells_.size()) {
            throw std::runtime_error("MeshState: numero di celle deal.II incoerente.");
        }
        for (unsigned int v = 0; v < 3; ++v) {
            const auto py_v = static_cast<std::size_t>(cells_[cell_idx][v]);
            const auto deal_v = static_cast<std::size_t>(cell->vertex_index(v));

            if (py_to_deal_vertex_[py_v] != std::numeric_limits<std::size_t>::max() &&
                py_to_deal_vertex_[py_v] != deal_v) {
                throw std::runtime_error("MeshState: conflitto nella mappa py_to_deal_vertex.");
            }
            if (deal_to_py_vertex_[deal_v] != std::numeric_limits<std::size_t>::max() &&
                deal_to_py_vertex_[deal_v] != py_v) {
                throw std::runtime_error("MeshState: conflitto nella mappa deal_to_py_vertex.");
            }
            py_to_deal_vertex_[py_v] = deal_v;
            deal_to_py_vertex_[deal_v] = py_v;
        }
        ++cell_idx;
    }

    py_to_dof_.assign(n_nodes, dealii::numbers::invalid_dof_index);
    dof_to_py_.assign(n_nodes, std::numeric_limits<std::size_t>::max());

    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        for (unsigned int v = 0; v < 3; ++v) {
            const auto deal_v = static_cast<std::size_t>(cell->vertex_index(v));
            const auto py_v = deal_to_py_vertex_.at(deal_v);
            const auto dof_idx = cell->vertex_dof_index(v, 0);
            py_to_dof_[py_v] = dof_idx;
            dof_to_py_[dof_idx] = py_v;
        }
    }

    verify_vertex_and_dof_mapping();
}

void MeshState::verify_vertex_and_dof_mapping() const {
    const std::size_t n_nodes = nodes_.size();
    std::vector<bool> seen_dof(n_nodes, false);

    for (std::size_t py_i = 0; py_i < n_nodes; ++py_i) {
        const auto deal_v = py_to_deal_vertex_.at(py_i);
        const auto dof = py_to_dof_.at(py_i);
        if (deal_v == std::numeric_limits<std::size_t>::max() ||
            dof == dealii::numbers::invalid_dof_index || dof >= n_nodes) {
            throw std::runtime_error("MeshState: associazione incompleta tra vertici Python e DoF deal.II.");
        }
        if (deal_to_py_vertex_.at(deal_v) != py_i || dof_to_py_.at(dof) != py_i) {
            throw std::runtime_error("MeshState: mappe vertici/DoF non biunivoche.");
        }
        if (seen_dof[dof]) {
            throw std::runtime_error("MeshState: DoF duplicato rilevato.");
        }
        seen_dof[dof] = true;
    }

    // Verifica esplicita delle coordinate fisiche di ogni vertice di cella deal.II rispetto a nodes_
    std::size_t cell_idx = 0;
    for (const auto& cell : dof_handler_.active_cell_iterators()) {
        for (unsigned int v = 0; v < 3; ++v) {
            const auto deal_v = static_cast<std::size_t>(cell->vertex_index(v));
            const auto py_v = deal_to_py_vertex_.at(deal_v);
            if (py_v != static_cast<std::size_t>(cells_[cell_idx][v])) {
                throw std::runtime_error("MeshState: disallineamento tra connettivita' Python e cella deal.II.");
            }
            const auto& pt = cell->vertex(v);
            const double dist = std::hypot(pt[0] - nodes_[py_v][0], pt[1] - nodes_[py_v][1]);
            if (dist > 1.0e-13) {
                throw std::runtime_error("MeshState: coordinata del vertice deal.II non coincide con il nodo Python.");
            }
            if (cell->vertex_dof_index(v, 0) != py_to_dof_[py_v]) {
                throw std::runtime_error("MeshState: incoerenza tra vertex_dof_index e py_to_dof.");
            }
        }
        ++cell_idx;
    }
}

void MeshState::sync_dealii_vertices() {
    for (const auto& cell : tria_.active_cell_iterators()) {
        for (unsigned int v = 0; v < 3; ++v) {
            const auto deal_v = static_cast<std::size_t>(cell->vertex_index(v));
            const auto py_v = deal_to_py_vertex_[deal_v];
            cell->vertex(v) = dealii::Point<2>(nodes_[py_v][0], nodes_[py_v][1]);
        }
    }
    verify_vertex_and_dof_mapping();
    ++geometry_revision_;
}

void MeshState::update_coordinates(const std::vector<Coord2D>& new_nodes, bool enforce_fixed_boundary) {
    if (new_nodes.size() != nodes_.size()) {
        throw std::invalid_argument("MeshState::update_coordinates: dimensione incompatibile.");
    }

    // Prepara un candidato separato: nessuna coordinata interna viene modificata
    // finché shape, finitezza e orientamento di tutti gli elementi non sono validi.
    std::vector<Coord2D> candidate = new_nodes;
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        if (!std::isfinite(candidate[i][0]) || !std::isfinite(candidate[i][1])) {
            throw std::invalid_argument("MeshState::update_coordinates: coordinate non finite.");
        }
        if (enforce_fixed_boundary && is_boundary_[i]) {
            candidate[i] = nodes_[i];
        }
    }

    constexpr double min_det_tol = 1.0e-15;
    for (std::size_t e = 0; e < cells_.size(); ++e) {
        const auto i0 = static_cast<std::size_t>(cells_[e][0]);
        const auto i1 = static_cast<std::size_t>(cells_[e][1]);
        const auto i2 = static_cast<std::size_t>(cells_[e][2]);
        const double det_j =
            (candidate[i1][0] - candidate[i0][0]) * (candidate[i2][1] - candidate[i0][1]) -
            (candidate[i2][0] - candidate[i0][0]) * (candidate[i1][1] - candidate[i0][1]);
        if (!std::isfinite(det_j) || det_j <= min_det_tol) {
            throw std::invalid_argument(
                "MeshState::update_coordinates: aggiornamento rifiutato; elemento invertito o degenere (cell_index=" +
                std::to_string(e) + ", det_J=" + std::to_string(det_j) + ").");
        }
    }

    const auto previous_nodes = nodes_;
    const auto previous_revision = geometry_revision_;
    nodes_ = std::move(candidate);
    try {
        sync_dealii_vertices();
    } catch (...) {
        nodes_ = previous_nodes;
        try {
            sync_dealii_vertices();
        } catch (...) {
            // Preserva l'eccezione originale; un errore nel rollback indicherebbe
            // una violazione interna più grave e non può essere mascherato.
        }
        geometry_revision_ = previous_revision;
        throw;
    }
}

void MeshState::apply_displacement(const std::vector<Coord2D>& displacement, bool enforce_fixed_boundary) {
    if (displacement.size() != nodes_.size()) {
        throw std::invalid_argument("MeshState::apply_displacement: dimensione incompatibile.");
    }
    std::vector<Coord2D> updated(nodes_.size());
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        if (!std::isfinite(displacement[i][0]) || !std::isfinite(displacement[i][1])) {
            throw std::invalid_argument("MeshState::apply_displacement: spostamento non finito.");
        }
        if (enforce_fixed_boundary && is_boundary_[i]) {
            updated[i] = nodes_[i];
        } else {
            updated[i] = {nodes_[i][0] + displacement[i][0], nodes_[i][1] + displacement[i][1]};
        }
    }
    update_coordinates(updated, false);
}

MeshState MeshState::cloned_with_coordinates(const std::vector<Coord2D>& new_nodes,
                                             bool enforce_fixed_boundary) const {
    MeshState copy(*this);
    copy.update_coordinates(new_nodes, enforce_fixed_boundary);
    return copy;
}

bool MeshState::has_same_topology(const MeshState& other) const noexcept {
    return (nodes_.size() == other.nodes_.size()) &&
           (cells_ == other.cells_) &&
           (boundary_nodes_ == other.boundary_nodes_);
}

std::size_t MeshState::py_node_to_deal_vertex(std::size_t py_idx) const {
    return py_to_deal_vertex_.at(py_idx);
}

std::size_t MeshState::deal_vertex_to_py_node(std::size_t deal_v_idx) const {
    return deal_to_py_vertex_.at(deal_v_idx);
}

dealii::types::global_dof_index MeshState::py_node_to_dof(std::size_t py_idx) const {
    return py_to_dof_.at(py_idx);
}

std::size_t MeshState::dof_to_py_node(dealii::types::global_dof_index dof_idx) const {
    return dof_to_py_.at(dof_idx);
}

} // namespace gdm