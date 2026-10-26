cmake_minimum_required(VERSION 3.20)

# deal.II inizializza le proprie impostazioni prima della chiamata a project().
if(NOT DEAL_II_DIR AND DEFINED ENV{DEAL_II_DIR})
    set(DEAL_II_DIR "$ENV{DEAL_II_DIR}" CACHE PATH "Directory di installazione di deal.II")
endif()
find_package(deal.II 9.7.1 REQUIRED HINTS "${DEAL_II_DIR}" "$ENV{DEAL_II_DIR}")
deal_ii_initialize_cached_variables()

project(gdm_cpp_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(GDM_CORE_SOURCES
    cpp_core/src/mesh_state.cpp
    cpp_core/src/mesh_metrics.cpp
    cpp_core/src/safety_filter.cpp
    cpp_core/src/pde_model.cpp
    cpp_core/src/fem_solver.cpp
    cpp_core/src/ale_solver.cpp
    cpp_core/src/remap.cpp
    cpp_core/src/diagnostics.cpp
    cpp_core/src/mms.cpp
)

add_library(gdm_numeric_core STATIC ${GDM_CORE_SOURCES})
target_include_directories(gdm_numeric_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp_core/include>
)
deal_ii_setup_target(gdm_numeric_core)

pybind11_add_module(_gdm_core MODULE cpp_core/bindings/pybind_module.cpp)
target_link_libraries(_gdm_core PRIVATE gdm_numeric_core)
# deal_ii_setup_target(_gdm_core) omesso per evitare il conflitto di firme target_link_libraries con pybind11.

# Modalità editable: l'estensione si trova accanto a python/gdm/__init__.py.
set_target_properties(_gdm_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/gdm"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/gdm"
)

# Installazione CMake convenzionale (la destinazione è relativa al prefix, non ai sorgenti).
install(TARGETS _gdm_core
    LIBRARY DESTINATION gdm
    RUNTIME DESTINATION gdm
)

enable_testing()

set(GDM_CPP_TESTS
    test_mesh_state
    test_mesh_metrics
    test_safety_filter
    test_fem_mms
    test_fem_convergence
    test_ale_gcl
    test_ale_convergence
)

foreach(test_name IN LISTS GDM_CPP_TESTS)
    add_executable(${test_name} "cpp_core/tests/${test_name}.cpp")
    target_link_libraries(${test_name} gdm_numeric_core)
    deal_ii_setup_target(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
