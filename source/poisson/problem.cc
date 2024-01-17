// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


#include <deal.II/fe/fe_q.h>

#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <factory.h>
#include <global.h>
#include <linear_algebra.h>
#include <log.h>
#include <poisson/matrixbased_operator.h>
#include <poisson/matrixfree_operator.h>
#include <poisson/problem.h>
#include <poisson/solvers.h>

#include <ctime>
#include <iomanip>
#include <sstream>

using namespace dealii;


template <int dim, typename LinearAlgebra, int spacedim = dim, typename... Args>
static std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>
create_operator(const std::string type, Args &&...args)
{
  if (type == "MatrixBased")
    {
      return std::make_unique<PoissonMatrixBased::PoissonOperator<dim, LinearAlgebra, spacedim>>(
        std::forward<Args>(args)...);
    }
  else if (type == "MatrixFree")
    {
      if constexpr (std::is_same_v<LinearAlgebra, dealiiTrilinos>)
        {
          return std::make_unique<PoissonMatrixFree::PoissonOperator<dim, LinearAlgebra, spacedim>>(
            std::forward<Args>(args)...);
        }
      else
        {
          AssertThrow(false, ExcMessage("MatrixFree only available with dealii & Trilinos!"));
        }
    }

  AssertThrow(false, ExcNotImplemented());
  return std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>();
}



namespace Poisson
{
  template <int dim, typename LinearAlgebra, int spacedim>
  Problem<dim, LinearAlgebra, spacedim>::Problem(const Parameter &prm)
    : mpi_communicator(MPI_COMM_WORLD)
    , prm(prm)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , dof_handler(triangulation)
  {
    TimerOutput::Scope t(getTimer(), "initialize_problem");

    // prepare name for logfile
    {
      time_t             now = time(nullptr);
      tm                *ltm = localtime(&now);
      std::ostringstream oss;
      oss << prm.file_stem << "-" << std::put_time(ltm, "%Y%m%d-%H%M%S");
      filename_stem = oss.str();
      filename_log  = filename_stem + ".log";
    }

    // prepare collections
    mapping_collection.push_back(MappingQ1<dim, spacedim>());

    for (unsigned int degree = 1; degree <= prm.prm_adaptation.max_p_degree; ++degree)
      {
        fe_collection.push_back(FE_Q<dim, spacedim>(degree));
        quadrature_collection.push_back(QGauss<dim>(degree + 1));
      }

    const unsigned int min_fe_index = prm.prm_adaptation.min_p_degree - 1;
    fe_collection.set_hierarchy(
      /*next_index=*/
      [](const typename hp::FECollection<dim> &fe_collection,
         const unsigned int                    fe_index) -> unsigned int {
        return ((fe_index + 1) < fe_collection.size()) ? fe_index + 1 : fe_index;
      },
      /*previous_index=*/
      [min_fe_index](const typename hp::FECollection<dim> &,
                     const unsigned int fe_index) -> unsigned int {
        Assert(fe_index >= min_fe_index, ExcMessage("Finite element is not part of hierarchy!"));
        return (fe_index > min_fe_index) ? fe_index - 1 : fe_index;
      });

    // prepare operator
    poisson_operator = create_operator<dim, LinearAlgebra, spacedim>(prm.operator_type,
                                                                     mapping_collection,
                                                                     quadrature_collection,
                                                                     fe_collection);

    // choose functions
    if (prm.grid_type == "reentrant corner")
      {
        boundary_function = Factory::create_function<dim>("reentrant corner");
        solution_function = Factory::create_function<dim>("reentrant corner");
        // rhs_function      = Factory::create_function<dim>("zero");
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    // choose adaptation strategy
    adaptation_strategy = Factory::create_adaptation<dim, typename LinearAlgebra::Vector, spacedim>(
      prm.adaptation_type,
      prm.prm_adaptation,
      locally_relevant_solution,
      fe_collection,
      dof_handler,
      triangulation);

    // cell weighting
    if (prm.adaptation_type != "h")
      {
        cell_weights.reinit(dof_handler,
                            parallel::CellWeights<dim, spacedim>::ndofs_weighting(
                              {prm.prm_adaptation.weighting_factor,
                               prm.prm_adaptation.weighting_exponent}));
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::initialize_grid()
  {
    TimerOutput::Scope t(getTimer(), "initialize_grid");

    Factory::create_grid(prm.grid_type, triangulation);

    if (prm.resume_filename.compare("") != 0)
      {
        resume_from_checkpoint();
      }
    else
      {
        const unsigned int min_fe_index = prm.prm_adaptation.min_p_degree - 1;

        // first, connect fe_collection for CellWeights before refining
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_active_fe_index(min_fe_index);

        dof_handler.distribute_dofs(fe_collection);

        triangulation.refine_global(adaptation_strategy->get_n_initial_refinements());

        // second, set up correct FE indices for hpFull
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_active_fe_index(min_fe_index);

        dof_handler.distribute_dofs(fe_collection);
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::setup_system()
  {
    TimerOutput::Scope t(getTimer(), "setup_system");

    {
      TimerOutput::Scope t(getTimer(), "distribute_dofs");

      dof_handler.distribute_dofs(fe_collection);
      partitioning.reinit(dof_handler);
    }

    {
      TimerOutput::Scope t(getTimer(), "reinit_vectors");

      locally_relevant_solution.reinit(partitioning.get_owned_dofs(),
                                       partitioning.get_relevant_dofs(),
                                       mpi_communicator);
    }

    {
      TimerOutput::Scope t(getTimer(), "make_constraints");

      constraints.clear();
      constraints.reinit(partitioning.get_relevant_dofs());

      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      if (prm.grid_type == "reentrant corner")
        {
          VectorTools::interpolate_boundary_values(
            mapping_collection, dof_handler, 0, *boundary_function, constraints);
        }
      else
        {
          Assert(false, ExcNotImplemented());
        }

#if false
#ifdef DEBUG
      // We have not dealt with chains of constraints on ghost cells yet.
      // Thus, we are content with verifying their consistency for now.
      const std::vector<IndexSet> locally_owned_dofs_per_processor =
        Utilities::MPI::all_gather(mpi_communicator, dof_handler.locally_owned_dofs());

      IndexSet locally_active_dofs;
      DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);

      AssertThrow(constraints.is_consistent_in_parallel(locally_owned_dofs_per_processor,
                                                        locally_active_dofs,
                                                        mpi_communicator,
                                                        /*verbose=*/true),
                  ExcMessage("AffineConstraints object contains inconsistencies!"));
#endif
#endif
      constraints.close();
    }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::solve()
  {
    TimerOutput::Scope t(getTimer(), "solve");

    // We need to introduce a vector that does not contain all ghost elements.
    typename LinearAlgebra::Vector completely_distributed_solution;
    poisson_operator->initialize_dof_vector(completely_distributed_solution);

    SolverControl solver_control(system_rhs.size(),
                                 prm.solver_tolerance_factor * system_rhs.l2_norm());

    if (prm.solver_type == "AMG")
      {
        solve_amg<dim, LinearAlgebra, spacedim>(solver_control,
                                                *poisson_operator,
                                                completely_distributed_solution,
                                                system_rhs);
      }
    else if (prm.solver_type == "GMG")
      {
        if constexpr (std::is_same_v<LinearAlgebra, dealiiTrilinos>)
          {
            if (prm.prm_multigrid.smoother_preconditioner_type == "Extended Diagonal")
              {
                solve_gmg<PreconditionExtendedDiagonal<typename LinearAlgebra::Vector>, dim, LinearAlgebra, spacedim>(solver_control,
                                                        *poisson_operator,
                                                        completely_distributed_solution,
                                                        system_rhs,
                                                        prm.prm_multigrid,
                                                        /*boundary_values=*/mapping_collection,
                                                        quadrature_collection,
                                                        dof_handler,
                                                        filename_stem + "-mglevel-cycle_" +
                                                          std::to_string(cycle) + ".log");
              }
            else if (prm.prm_multigrid.smoother_preconditioner_type == "ASM")
              {
                solve_gmg<PreconditionASM<typename LinearAlgebra::Vector>, dim, LinearAlgebra, spacedim>(solver_control,
                                                          *poisson_operator,
                                                          completely_distributed_solution,
                                                          system_rhs,
                                                          prm.prm_multigrid,
                                                          /*boundary_values=*/mapping_collection,
                                                          quadrature_collection,
                                                          dof_handler,
                                                          filename_stem + "-mglevel-cycle_" +
                                                            std::to_string(cycle) + ".log");
              }
            else if (prm.prm_multigrid.smoother_preconditioner_type == "Diagonal")
              {
                solve_gmg<DiagonalMatrix<typename LinearAlgebra::Vector>, dim, LinearAlgebra, spacedim>(solver_control,
                                                          *poisson_operator,
                                                          completely_distributed_solution,
                                                          system_rhs,
                                                          prm.prm_multigrid,
                                                          /*boundary_values=*/mapping_collection,
                                                          quadrature_collection,
                                                          dof_handler,
                                                          filename_stem + "-mglevel-cycle_" +
                                                            std::to_string(cycle) + ".log");
              }
            else
              {
                AssertThrow(false, ExcNotImplemented());
              }
          }
        else
          {
            AssertThrow(false, ExcMessage("GMG is only available with dealii & Trilinos!"));
          }
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    Log::log_iterations(solver_control);

    constraints.distribute(completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
    locally_relevant_solution.update_ghost_values();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::compute_errors()
  {
    TimerOutput::Scope t(getTimer(), "compute_errors");

    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      locally_relevant_solution,
                                      *solution_function,
                                      difference_per_cell,
                                      quadrature_collection,
                                      VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      locally_relevant_solution,
                                      *solution_function,
                                      difference_per_cell,
                                      quadrature_collection,
                                      VectorTools::H1_norm);
    const double H1_error =
      VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::H1_norm);

    getPCOut() << "   L2 error:                     " << L2_error << std::endl
               << "   H1 error:                     " << H1_error << std::endl;

    TableHandler &table = getTable();
    table.add_value("L2", L2_error);
    table.add_value("H1", H1_error);
    table.set_scientific("L2", true);
    table.set_scientific("H1", true);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::output_results()
  {
    TimerOutput::Scope t(getTimer(), "output_results");

    Vector<float> fe_degrees(triangulation.n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        fe_degrees(cell->active_cell_index()) = cell->get_fe().degree;

    // TODO: Also write out future fe degree
    Vector<float> future_fe_degrees(triangulation.n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        future_fe_degrees(cell->active_cell_index()) = fe_collection[cell->future_fe_index()].degree;

    Vector<float> subdomain(triangulation.n_active_cells());
    for (auto &subd : subdomain)
      subd = triangulation.locally_owned_subdomain();

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(locally_relevant_solution, "solution");
    data_out.add_data_vector(fe_degrees, "fe_degree");
    data_out.add_data_vector(future_fe_degrees, "future_fe_degree");
    data_out.add_data_vector(subdomain, "subdomain");

    if (adaptation_strategy->get_error_estimates().size() > 0)
      data_out.add_data_vector(adaptation_strategy->get_error_estimates(), "error");
    if (adaptation_strategy->get_hp_indicators().size() > 0)
      data_out.add_data_vector(adaptation_strategy->get_hp_indicators(), "hp_indicator");

    data_out.build_patches(mapping_collection);

    data_out.write_vtu_with_pvtu_record("./", prm.file_stem, cycle, mpi_communicator, 2, 1);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::resume_from_checkpoint()
  {
    // extract cycle from filename
    const auto pos = prm.resume_filename.rfind(".cycle-");
    AssertThrow(pos != std::string::npos,
                ExcMessage("Checkpoint: filename misses information about cycle!"));
    const auto substring = prm.resume_filename.substr(pos + 7);
    try
      {
        cycle = std::stoul(substring);
      }
    catch (...)
      {
        AssertThrow(false, ExcMessage("Checkpoint: invalid cycle!"));
      }

    triangulation.load(prm.resume_filename);

    // custom repartitioning using DoFs requires correctly assigned FEs
    dof_handler.deserialize_active_fe_indices();
    dof_handler.distribute_dofs(fe_collection);
    triangulation.repartition();

    // unpack after repartitioning to avoid unnecessary data transfer
    adaptation_strategy->unpack_after_serialization();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::write_to_checkpoint()
  {
    // write triangulation and data
    dof_handler.prepare_for_serialization_of_active_fe_indices();
    adaptation_strategy->prepare_for_serialization();

    const std::string filename =
      prm.file_stem + ".cycle-" + Utilities::to_string(cycle, 2) + ".checkpoint";
    triangulation.save(filename);

    getPCOut() << "Checkpoint written." << std::endl;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::run()
  {
    getTable().set_auto_fill_mode(true);

    for (cycle = 0; cycle < adaptation_strategy->get_n_cycles(); ++cycle)
      {
        {
          TimerOutput::Scope t(getTimer(), "full_cycle");

          if (cycle == 0)
            {
              initialize_grid();
            }
          else
            {
              adaptation_strategy->refine();

              if ((prm.checkpoint_frequency > 0) && (cycle % prm.checkpoint_frequency == 0))
                write_to_checkpoint();
            }

          Log::log_cycle(cycle, prm);

          setup_system();

          Log::log_hp_diagnostics(triangulation, dof_handler, constraints);

          poisson_operator->reinit(partitioning, dof_handler, constraints, system_rhs, nullptr);

          if (prm.operator_type == "MatrixBased" || prm.log_nonzero_elements)
            Log::log_nonzero_elements(poisson_operator->get_system_matrix());

          solve();

          compute_errors();
          adaptation_strategy->estimate_mark();

          if ((prm.output_frequency > 0) && (cycle % prm.output_frequency == 0))
            output_results();
        }

        Log::log_timing_statistics(mpi_communicator);

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          {
            std::ofstream logstream(filename_log);
            getTable().write_text(logstream);
          }

        getTimer().reset();
        getTable().start_new_row();
      }
  }



  // explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class Problem<2, dealiiTrilinos, 2>;
  template class Problem<3, dealiiTrilinos, 3>;
  template class Problem<2, Trilinos, 2>;
  template class Problem<3, Trilinos, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
  template class Problem<2, PETSc, 2>;
  template class Problem<3, PETSc, 3>;
#endif
} // namespace Poisson
