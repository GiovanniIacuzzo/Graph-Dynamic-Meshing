#include "gdm/ale_solver.hpp"
#include "gdm/diagnostics.hpp"
#include "gdm/fem_solver.hpp"
#include "gdm/mesh_metrics.hpp"
#include "gdm/mesh_state.hpp"
#include "gdm/mms.hpp"
#include "gdm/parameters.hpp"
#include "gdm/pde_model.hpp"
#include "gdm/remap.hpp"
#include "gdm/safety_filter.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace gdm::bindings {

std::vector<MeshState::Coord2D> array_to_coords(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& arr,
    const char* name) {
    if (arr.ndim() != 2 || arr.shape(1) != 2) {
        throw std::invalid_argument(std::string(name) + " deve avere forma (N, 2).");
    }
    const auto r = arr.unchecked<2>();
    const auto n = static_cast<std::size_t>(r.shape(0));
    std::vector<MeshState::Coord2D> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = r(i, 0);
        const double y = r(i, 1);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            throw std::invalid_argument(std::string(name) + " contiene valori NaN o Inf.");
        }
        out[i] = {x, y};
    }
    return out;
}

std::vector<MeshState::CellConn> array_to_cells(
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& arr) {
    if (arr.ndim() != 2 || arr.shape(1) != 3) {
        throw std::invalid_argument("cells deve avere forma (M, 3).");
    }
    const auto r = arr.unchecked<2>();
    const auto m = static_cast<std::size_t>(r.shape(0));
    std::vector<MeshState::CellConn> out(m);
    for (std::size_t e = 0; e < m; ++e) {
        out[e] = {r(e, 0), r(e, 1), r(e, 2)};
    }
    return out;
}

std::vector<double> array_to_scalar_vec(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& arr,
    const char* name) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " deve essere un vettore 1D di lunghezza N.");
    }
    const auto r = arr.unchecked<1>();
    const auto n = static_cast<std::size_t>(r.shape(0));
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(r(i))) {
            throw std::invalid_argument(std::string(name) + " contiene valori non finiti.");
        }
        out[i] = r(i);
    }
    return out;
}

py::array_t<double> coords_to_array(const std::vector<MeshState::Coord2D>& coords) {
    py::array_t<double> out({static_cast<py::ssize_t>(coords.size()), static_cast<py::ssize_t>(2)});
    auto w = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < coords.size(); ++i) {
        w(i, 0) = coords[i][0];
        w(i, 1) = coords[i][1];
    }
    return out;
}

py::array_t<std::int64_t> cells_to_array(const std::vector<MeshState::CellConn>& cells) {
    py::array_t<std::int64_t> out({static_cast<py::ssize_t>(cells.size()), static_cast<py::ssize_t>(3)});
    auto w = out.mutable_unchecked<2>();
    for (std::size_t e = 0; e < cells.size(); ++e) {
        w(e, 0) = cells[e][0];
        w(e, 1) = cells[e][1];
        w(e, 2) = cells[e][2];
    }
    return out;
}

py::array_t<std::int64_t> edges_to_array(const std::vector<MeshState::EdgePair>& edges) {
    py::array_t<std::int64_t> out({static_cast<py::ssize_t>(edges.size()), static_cast<py::ssize_t>(2)});
    auto w = out.mutable_unchecked<2>();
    for (std::size_t k = 0; k < edges.size(); ++k) {
        w(k, 0) = edges[k][0];
        w(k, 1) = edges[k][1];
    }
    return out;
}

py::array_t<double> scalar_vec_to_array(const std::vector<double>& v) {
    py::array_t<double> out(static_cast<py::ssize_t>(v.size()));
    auto w = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < v.size(); ++i) {
        w(i) = v[i];
    }
    return out;
}

py::array_t<bool> bool_vec_to_array(const std::vector<bool>& v) {
    py::array_t<bool> out(static_cast<py::ssize_t>(v.size()));
    auto w = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < v.size(); ++i) {
        w(i) = v[i];
    }
    return out;
}

} // namespace gdm::bindings

PYBIND11_MODULE(_gdm_core, m) {
    m.doc() = "Nucleo numerico C++20 / deal.II 9.8.0 per Graph-Dynamic-Meshing (FEM P1, ALE D-GCL, Safety Filter, Remap).";

    using namespace gdm;
    using namespace gdm::bindings;

    py::class_<DomainParameters>(m, "DomainParameters")
        .def(py::init<>())
        .def_readwrite("x_min", &DomainParameters::x_min)
        .def_readwrite("x_max", &DomainParameters::x_max)
        .def_readwrite("y_min", &DomainParameters::y_min)
        .def_readwrite("y_max", &DomainParameters::y_max)
        .def_readwrite("nx", &DomainParameters::nx)
        .def_readwrite("ny", &DomainParameters::ny)
        .def("validate", &DomainParameters::validate);

    py::class_<PDEParameters>(m, "PDEParameters")
        .def(py::init<>())
        .def_readwrite("nu", &PDEParameters::nu)
        .def_readwrite("bx", &PDEParameters::bx)
        .def_readwrite("by", &PDEParameters::by)
        .def_readwrite("sigma", &PDEParameters::sigma)
        .def_readwrite("beta", &PDEParameters::beta)
        .def_readwrite("T", &PDEParameters::T)
        .def("validate", &PDEParameters::validate);

    py::class_<SafetyParameters>(m, "SafetyParameters")
        .def(py::init<>())
        .def_readwrite("epsilon", &SafetyParameters::epsilon)
        .def_readwrite("tol_deg", &SafetyParameters::tol_deg)
        .def_readwrite("safety_margin", &SafetyParameters::safety_margin)
        .def_readwrite("lock_boundary_nodes", &SafetyParameters::lock_boundary_nodes)
        .def_readwrite("verbose", &SafetyParameters::verbose)
        .def("validate", &SafetyParameters::validate);

    py::class_<LinearSolverParameters>(m, "LinearSolverParameters")
        .def(py::init<>())
        .def_readwrite("rel_tol", &LinearSolverParameters::rel_tol)
        .def_readwrite("abs_tol", &LinearSolverParameters::abs_tol)
        .def_readwrite("max_iterations", &LinearSolverParameters::max_iterations)
        .def_readwrite("gmres_krylov_dim", &LinearSolverParameters::gmres_krylov_dim)
        .def_readwrite("ssor_omega", &LinearSolverParameters::ssor_omega)
        .def_readwrite("use_direct_solver", &LinearSolverParameters::use_direct_solver)
        .def_readwrite("right_preconditioning", &LinearSolverParameters::right_preconditioning)
        .def_readwrite("ale_rhs_norm_floor", &LinearSolverParameters::ale_rhs_norm_floor)
        .def("validate", &LinearSolverParameters::validate);

    py::class_<StabilizationParameters>(m, "StabilizationParameters")
        .def(py::init<>())
        .def_readwrite("use_supg", &StabilizationParameters::use_supg)
        .def_readwrite("c_time", &StabilizationParameters::c_time)
        .def_readwrite("c_advection", &StabilizationParameters::c_advection)
        .def_readwrite("c_diffusion", &StabilizationParameters::c_diffusion)
        .def("validate", &StabilizationParameters::validate);

    py::class_<MeshState>(m, "MeshState")
        .def(py::init([](const py::array_t<double, py::array::c_style | py::array::forcecast>& nodes,
                         const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& cells,
                         const std::vector<std::int64_t>& boundary_nodes) {
                 auto c_nodes = array_to_coords(nodes, "nodes");
                 auto c_cells = array_to_cells(cells);
                 py::gil_scoped_release release;
                 return MeshState(c_nodes, c_cells, boundary_nodes);
             }),
             py::arg("nodes"),
             py::arg("cells"),
             py::arg("boundary_nodes") = std::vector<std::int64_t>{})
        .def_static("create_structured_mesh", &MeshState::create_structured_mesh, py::arg("domain"))
        .def_property_readonly("num_nodes", &MeshState::num_nodes)
        .def_property_readonly("num_cells", &MeshState::num_cells)
        .def_property_readonly("nodes", [](const MeshState& self) { return coords_to_array(self.nodes()); })
        .def_property_readonly("cells", [](const MeshState& self) { return cells_to_array(self.cells()); })
        .def_property_readonly("boundary_nodes", [](const MeshState& self) {
            // Allocazione proprietaria: l'array NumPy non conserva un puntatore a memoria C++ temporanea.
            py::array_t<std::int64_t> out(static_cast<py::ssize_t>(self.boundary_nodes().size()));
            auto w = out.mutable_unchecked<1>();
            for (std::size_t i = 0; i < self.boundary_nodes().size(); ++i)
                w(i) = self.boundary_nodes()[i];
            return out;
        })
        .def_property_readonly("edges", [](const MeshState& self) { return edges_to_array(self.directed_edges()); })
        .def_property_readonly("directed_edges", [](const MeshState& self) { return edges_to_array(self.directed_edges()); })
        .def_property_readonly("undirected_edges", [](const MeshState& self) { return edges_to_array(self.undirected_edges()); })
        .def_property_readonly("geometry_revision", &MeshState::geometry_revision)
        .def("update_coordinates",
             [](MeshState& self,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& new_nodes,
                bool enforce_fixed_boundary) {
                 auto coords = array_to_coords(new_nodes, "new_nodes");
                 py::gil_scoped_release release;
                 self.update_coordinates(coords, enforce_fixed_boundary);
             },
             py::arg("new_nodes"),
             py::arg("enforce_fixed_boundary") = false)
        .def("apply_displacement",
             [](MeshState& self,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& disp,
                bool enforce_fixed_boundary) {
                 auto d = array_to_coords(disp, "displacement");
                 py::gil_scoped_release release;
                 self.apply_displacement(d, enforce_fixed_boundary);
             },
             py::arg("displacement"),
             py::arg("enforce_fixed_boundary") = true)
        .def("cloned_with_coordinates",
             [](const MeshState& self,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& new_nodes,
                bool enforce_fixed_boundary) {
                 auto coords = array_to_coords(new_nodes, "new_nodes");
                 py::gil_scoped_release release;
                 return self.cloned_with_coordinates(coords, enforce_fixed_boundary);
             },
             py::arg("new_nodes"),
             py::arg("enforce_fixed_boundary") = false);

    py::class_<ManufacturedConvectionDiffusion2D>(m, "ManufacturedConvectionDiffusion2D")
        .def(py::init<const PDEParameters&>(), py::arg("params") = PDEParameters{})
        .def_property_readonly("bx", [](const ManufacturedConvectionDiffusion2D& self) { return self.parameters().bx; })
        .def_property_readonly("by", [](const ManufacturedConvectionDiffusion2D& self) { return self.parameters().by; })
        .def_property_readonly("nu", [](const ManufacturedConvectionDiffusion2D& self) { return self.parameters().nu; })
        .def_property_readonly("sigma", [](const ManufacturedConvectionDiffusion2D& self) { return self.parameters().sigma; })
        .def_property_readonly("beta", [](const ManufacturedConvectionDiffusion2D& self) { return self.parameters().beta; })
        .def_property_readonly("T", [](const ManufacturedConvectionDiffusion2D& self) { return self.parameters().T; })
        .def("get_center", &ManufacturedConvectionDiffusion2D::get_center, py::arg("t"))
        // Array API used by the native runtime: nodes has shape (N, 2).
        .def("evaluate_solution",
             [](const ManufacturedConvectionDiffusion2D& self,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& nodes,
                double t) {
                 auto coords = array_to_coords(nodes, "nodes");
                 std::vector<double> res;
                 {
                     py::gil_scoped_release release;
                     res = self.evaluate_solution_on_nodes(coords, t);
                 }
                 return scalar_vec_to_array(res);
             },
             py::arg("nodes"),
             py::arg("t"))
        // Compatibility array API for oracle/objective code: x, y are matching 1D arrays.
        .def("evaluate_solution",
             [](const ManufacturedConvectionDiffusion2D& self,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& x,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& y,
                double t) {
                 if (x.ndim() != 1 || y.ndim() != 1 || x.shape(0) != y.shape(0))
                     throw std::invalid_argument("x e y devono essere vettori 1D della stessa lunghezza.");
                 const auto xr = x.unchecked<1>();
                 const auto yr = y.unchecked<1>();
                 std::vector<double> res(static_cast<std::size_t>(x.shape(0)));
                 for (py::ssize_t i = 0; i < x.shape(0); ++i) {
                     if (!std::isfinite(xr(i)) || !std::isfinite(yr(i)))
                         throw std::invalid_argument("x o y contiene valori non finiti.");
                     res[static_cast<std::size_t>(i)] = self.evaluate_solution(xr(i), yr(i), t);
                 }
                 return scalar_vec_to_array(res);
             },
             py::arg("x"), py::arg("y"), py::arg("t"))
        .def("evaluate_exact",
             [](const ManufacturedConvectionDiffusion2D& self,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& x,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& y,
                double t) {
                 if (x.ndim() != 1 || y.ndim() != 1 || x.shape(0) != y.shape(0))
                     throw std::invalid_argument("x e y devono essere vettori 1D della stessa lunghezza.");
                 const auto xr = x.unchecked<1>();
                 const auto yr = y.unchecked<1>();
                 const auto n = static_cast<std::size_t>(x.shape(0));
                 std::vector<double> u(n), ux(n), uy(n), uxx(n), uyy(n), ut(n), forcing(n);
                 for (std::size_t i = 0; i < n; ++i) {
                     const double xx = xr(static_cast<py::ssize_t>(i));
                     const double yy = yr(static_cast<py::ssize_t>(i));
                     if (!std::isfinite(xx) || !std::isfinite(yy))
                         throw std::invalid_argument("x o y contiene valori non finiti.");
                     const auto e = self.evaluate_exact(xx, yy, t);
                     u[i] = e.u; ux[i] = e.ux; uy[i] = e.uy;
                     uxx[i] = e.uxx; uyy[i] = e.uyy; ut[i] = e.ut; forcing[i] = e.f;
                 }
                 return py::make_tuple(scalar_vec_to_array(u), scalar_vec_to_array(ux),
                                       scalar_vec_to_array(uy), scalar_vec_to_array(uxx),
                                       scalar_vec_to_array(uyy), scalar_vec_to_array(ut),
                                       scalar_vec_to_array(forcing));
             },
             py::arg("x"), py::arg("y"), py::arg("t"));

    py::class_<ConvectionDiffusionFEM2D>(m, "ConvectionDiffusionFEM2D", py::dynamic_attr())
        .def(py::init<const MeshState&, const ManufacturedConvectionDiffusion2D&,
                      const LinearSolverParameters&, const StabilizationParameters&>(),
             py::arg("mesh"),
             py::arg("pde"),
             py::arg("solver_params") = LinearSolverParameters{},
             py::arg("stabilization_params") = StabilizationParameters{},
             py::keep_alive<1, 2>())
        .def("rebuild_for_mesh",
             [](py::object self_obj, py::object mesh_obj) {
                 auto& self = self_obj.cast<ConvectionDiffusionFEM2D&>();
                 const auto& mesh = mesh_obj.cast<const MeshState&>();
                 {
                     py::gil_scoped_release release;
                     self.rebuild_for_mesh(mesh);
                 }
                 // ConvectionDiffusionFEM2D conserva un puntatore alla mesh: trattieni
                 // esplicitamente l'ultima mesh associata senza accumulare tutte le mesh
                 // storiche a ogni remeshing.
                 self_obj.attr("_gdm_mesh_keepalive") = mesh_obj;
             },
             py::arg("mesh"))
        .def("step_backward_euler",
             [](const ConvectionDiffusionFEM2D& self,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& u_n,
                double dt,
                double t_next) {
                 auto u_vec = array_to_scalar_vec(u_n, "u_n");
                 std::vector<double> u_next;
                 {
                     py::gil_scoped_release release;
                     u_next = self.step_backward_euler(u_vec, dt, t_next);
                 }
                 return scalar_vec_to_array(u_next);
             },
             py::arg("u_n"),
             py::arg("dt"),
             py::arg("t_next"))
        .def("step_backward_euler_with_diagnostics",
             [](const ConvectionDiffusionFEM2D& self,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& u_n,
                double dt,
                double t_next) {
                 auto u_vec = array_to_scalar_vec(u_n, "u_n");
                 std::vector<double> u_next;
                 double residual = 0.0, assembly_seconds = 0.0, solve_seconds = 0.0;
                 double rhs_norm = 0.0, solver_tolerance = 0.0;
                 {
                     py::gil_scoped_release release;
                     u_next = self.step_backward_euler(
                         u_vec, dt, t_next, &residual, &assembly_seconds, &solve_seconds,
                         &rhs_norm, &solver_tolerance);
                 }
                 py::dict result;
                 result["u_next"] = scalar_vec_to_array(u_next);
                 result["linear_solver_residual"] = residual;
                 result["assembly_seconds"] = assembly_seconds;
                 result["solve_seconds"] = solve_seconds;
                 result["linear_solver_rhs_norm"] = rhs_norm;
                 result["linear_solver_tolerance"] = solver_tolerance;
                 return result;
             },
             py::arg("u_n"), py::arg("dt"), py::arg("t_next"))
        .def("compute_discrete_mass",
             [](const ConvectionDiffusionFEM2D& self,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& u) {
                 auto u_vec = array_to_scalar_vec(u, "u");
                 py::gil_scoped_release release;
                 return self.compute_discrete_mass(u_vec);
             },
             py::arg("u"));

    py::class_<ALEDiagnostics>(m, "ALEDiagnostics")
        .def_readonly("max_elem_gcl_residual", &ALEDiagnostics::max_elem_gcl_residual)
        .def_readonly("algebraic_gcl_residual_linf", &ALEDiagnostics::algebraic_gcl_residual_linf)
        .def_readonly("dual_c_mesh_col_sum_linf", &ALEDiagnostics::dual_c_mesh_col_sum_linf)
        .def_readonly("discrete_mass_n", &ALEDiagnostics::discrete_mass_n)
        .def_readonly("discrete_mass_next", &ALEDiagnostics::discrete_mass_next)
        .def_readonly("linear_solver_residual", &ALEDiagnostics::linear_solver_residual)
        .def_readonly("linear_solver_rhs_norm", &ALEDiagnostics::linear_solver_rhs_norm)
        .def_readonly("linear_solver_tolerance", &ALEDiagnostics::linear_solver_tolerance)
        .def_readonly("element_gcl_residuals", &ALEDiagnostics::element_gcl_residuals);

    py::class_<ALESolver2D>(m, "ALESolver2D")
        .def(py::init<const ManufacturedConvectionDiffusion2D*, double, double, double,
                      const LinearSolverParameters&, const StabilizationParameters&>(),
             py::arg("pde") = nullptr,
             py::arg("nu") = 1.0e-3,
             py::arg("bx") = 0.4,
             py::arg("by") = 0.2,
             py::arg("solver_params") = LinearSolverParameters{},
             py::arg("stabilization_params") = StabilizationParameters{},
             py::keep_alive<1, 2>())
        .def("step_ale",
             [](const ALESolver2D& self,
                const MeshState& mesh_n,
                const MeshState& mesh_next,
                const py::array_t<double, py::array::c_style | py::array::forcecast>& u_n,
                double dt,
                double t_next,
                bool pure_gcl,
                bool enforce_dirichlet) {
                 auto u_vec = array_to_scalar_vec(u_n, "u_n");
                 ALEStepResult res;
                 {
                     py::gil_scoped_release release;
                     res = self.step_ale(mesh_n, mesh_next, u_vec, dt, t_next, pure_gcl, enforce_dirichlet);
                 }
                 return py::make_tuple(scalar_vec_to_array(res.u_next), res.diagnostics);
             },
             py::arg("mesh_n"),
             py::arg("mesh_next"),
             py::arg("u_n"),
             py::arg("dt"),
             py::arg("t_next"),
             py::arg("pure_gcl") = false,
             py::arg("enforce_dirichlet") = true);

    m.def("compute_mesh_metrics_arrays",
          [](const py::array_t<double, py::array::c_style | py::array::forcecast>& nodes,
             const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& cells,
             double tol_degenerate) {
              const auto c_nodes = array_to_coords(nodes, "nodes");
              const auto c_cells = array_to_cells(cells);
              MeshQualityMetrics m_out;
              {
                  py::gil_scoped_release release;
                  m_out = compute_mesh_metrics(c_nodes, c_cells, tol_degenerate);
              }
              py::dict d;
              d["jacobians"] = scalar_vec_to_array(m_out.jacobians);
              d["areas"] = scalar_vec_to_array(m_out.areas);
              d["normalized_jacobians"] = scalar_vec_to_array(m_out.normalized_jacobians);
              d["aspect_ratio"] = scalar_vec_to_array(m_out.aspect_ratio);
              d["min_angle"] = scalar_vec_to_array(m_out.min_angle);
              d["max_angle"] = scalar_vec_to_array(m_out.max_angle);
              d["escobar_quality"] = scalar_vec_to_array(m_out.escobar_quality);
              d["inverted"] = bool_vec_to_array(m_out.inverted);
              d["degenerate"] = bool_vec_to_array(m_out.degenerate);
              d["min_jacobian"] = m_out.min_jacobian;
              d["max_jacobian"] = m_out.max_jacobian;
              d["min_escobar_quality"] = m_out.min_escobar_quality;
              d["max_aspect_ratio"] = m_out.max_aspect_ratio;
              d["mean_escobar_quality"] = m_out.mean_escobar_quality;
              d["num_inverted"] = m_out.num_inverted;
              d["num_degenerate"] = m_out.num_degenerate;
              return d;
          },
          py::arg("nodes"), py::arg("cells"), py::arg("tol_degenerate") = 1.0e-14);

    m.def("compute_mesh_metrics",
          [](const MeshState& mesh, double tol_degenerate) {
              MeshQualityMetrics m_out;
              {
                  py::gil_scoped_release release;
                  m_out = compute_mesh_metrics(mesh, tol_degenerate);
              }
              py::dict d;
              d["jacobians"] = scalar_vec_to_array(m_out.jacobians);
              d["areas"] = scalar_vec_to_array(m_out.areas);
              d["normalized_jacobians"] = scalar_vec_to_array(m_out.normalized_jacobians);
              d["aspect_ratio"] = scalar_vec_to_array(m_out.aspect_ratio);
              d["min_angle"] = scalar_vec_to_array(m_out.min_angle);
              d["max_angle"] = scalar_vec_to_array(m_out.max_angle);
              d["escobar_quality"] = scalar_vec_to_array(m_out.escobar_quality);
              d["inverted"] = bool_vec_to_array(m_out.inverted);
              d["degenerate"] = bool_vec_to_array(m_out.degenerate);
              d["min_jacobian"] = m_out.min_jacobian;
              d["max_jacobian"] = m_out.max_jacobian;
              d["min_escobar_quality"] = m_out.min_escobar_quality;
              d["max_aspect_ratio"] = m_out.max_aspect_ratio;
              d["mean_escobar_quality"] = m_out.mean_escobar_quality;
              d["num_inverted"] = m_out.num_inverted;
              d["num_degenerate"] = m_out.num_degenerate;
              return d;
          },
          py::arg("mesh"),
          py::arg("tol_degenerate") = 1.0e-14);

    m.def("apply_safety_filter",
          [](const MeshState& mesh,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& displacement,
             const SafetyParameters& params) {
              auto disp = array_to_coords(displacement, "displacement");
              SafetyFilterResult res;
              {
                  py::gil_scoped_release release;
                  res = apply_safety_filter(mesh, disp, params);
              }
              py::dict d;
              d["alpha_global"] = res.alpha_global;
              d["safe_nodes"] = coords_to_array(res.safe_nodes);
              d["is_safe"] = res.is_safe;
              d["min_jacobian_before"] = res.min_jacobian_before;
              d["min_jacobian_after"] = res.min_jacobian_after;
              d["limiting_cell_index"] = res.limiting_cell_index;
              d["filter_activated"] = res.filter_activated;
              return d;
          },
          py::arg("mesh"),
          py::arg("displacement"),
          py::arg("params") = SafetyParameters{});

    m.def("compute_field_gradients",
          [](const MeshState& mesh,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& u,
             const ManufacturedConvectionDiffusion2D* pde,
             double t,
             bool exact_gradient) {
              auto u_vec = array_to_scalar_vec(u, "u");
              std::vector<MeshState::Coord2D> grad;
              {
                  py::gil_scoped_release release;
                  grad = compute_field_gradients(mesh, u_vec, pde, t, exact_gradient);
              }
              return coords_to_array(grad);
          },
          py::arg("mesh"),
          py::arg("u"),
          py::arg("pde") = nullptr,
          py::arg("t") = 0.0,
          py::arg("exact_gradient") = false);

    m.def("remap_p1_solution",
          [](const MeshState& old_mesh,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& old_u,
             const MeshState& new_mesh) {
              auto u_vec = array_to_scalar_vec(old_u, "old_u");
              RemapDiagnostics res;
              {
                  py::gil_scoped_release release;
                  res = remap_p1_solution(old_mesh, u_vec, new_mesh);
              }
              py::dict d;
              d["remapped_field"] = scalar_vec_to_array(res.remapped_field);
              d["mass_before"] = res.mass_before;
              d["mass_after"] = res.mass_after;
              d["relative_mass_drift"] = res.relative_mass_drift;
              d["elapsed_ms"] = res.elapsed_ms;
              return d;
          },
          py::arg("old_mesh"),
          py::arg("old_u"),
          py::arg("new_mesh"));

    m.def("remap_l2_projection",
          [](const MeshState& old_mesh,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& old_u,
             const MeshState& new_mesh,
             const LinearSolverParameters& solver_params) {
              auto u_vec = array_to_scalar_vec(old_u, "old_u");
              RemapDiagnostics res;
              {
                  py::gil_scoped_release release;
                  res = remap_l2_projection(old_mesh, u_vec, new_mesh, solver_params);
              }
              py::dict d;
              d["remapped_field"] = scalar_vec_to_array(res.remapped_field);
              d["mass_before"] = res.mass_before;
              d["mass_after"] = res.mass_after;
              d["relative_mass_drift"] = res.relative_mass_drift;
              d["elapsed_ms"] = res.elapsed_ms;
              return d;
          },
          py::arg("old_mesh"),
          py::arg("old_u"),
          py::arg("new_mesh"),
          py::arg("solver_params") = LinearSolverParameters{});

    m.def("compute_continuous_errors",
          [](const MeshState& mesh,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& u_h,
             const ManufacturedConvectionDiffusion2D& pde,
             double t) {
              auto u_vec = array_to_scalar_vec(u_h, "u_h");
              ContinuousErrorNorms norms;
              {
                  py::gil_scoped_release release;
                  norms = compute_continuous_errors(mesh, u_vec, pde, t);
              }
              return py::make_tuple(norms.rel_l2_error, norms.rel_h1_seminorm_error);
          },
          py::arg("mesh"),
          py::arg("u_h"),
          py::arg("pde"),
          py::arg("t"));

    // Full error details for reporting utilities that require absolute and relative norms.
    m.def("compute_continuous_error_details",
          [](const MeshState& mesh,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& u_h,
             const ManufacturedConvectionDiffusion2D& pde,
             double t) {
              auto u_vec = array_to_scalar_vec(u_h, "u_h");
              ContinuousErrorNorms norms;
              {
                  py::gil_scoped_release release;
                  norms = compute_continuous_errors(mesh, u_vec, pde, t);
              }
              py::dict out;
              out["abs_l2_error"] = norms.abs_l2_error;
              out["rel_l2_error"] = norms.rel_l2_error;
              out["abs_h1_seminorm_error"] = norms.abs_h1_seminorm_error;
              out["rel_h1_seminorm_error"] = norms.rel_h1_seminorm_error;
              return out;
          },
          py::arg("mesh"),
          py::arg("u_h"),
          py::arg("pde"),
          py::arg("t"));

    m.def("compute_discrete_mass",
          [](const MeshState& mesh,
             const py::array_t<double, py::array::c_style | py::array::forcecast>& u) {
              auto u_vec = array_to_scalar_vec(u, "u");
              py::gil_scoped_release release;
              return ALESolver2D::compute_discrete_mass(mesh, u_vec);
          },
          py::arg("mesh"), py::arg("u"));
}