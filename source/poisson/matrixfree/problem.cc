// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2022 by the deal.II authors
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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <base/global.h>
#include <base/linear_algebra.h>
#include <base/log.h>
#include <factory.h>
#include <poisson/amg.h>
#include <poisson/gmg.h>
#include <poisson/matrixbased/poisson_operator.h>
#include <poisson/matrixfree/poisson_operator.h>
#include <poisson/problem.h>

#include <ctime>
#include <iomanip>
#include <sstream>

using namespace dealii;


namespace
{
  template <int dim, typename LinearAlgebra, int spacedim = dim, typename... Args>
  std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>
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
            return std::make_unique<
              PoissonMatrixFree::PoissonOperator<dim, LinearAlgebra, spacedim>>(
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
} // namespace



namespace Poisson
{
  template <int dim, typename LinearAlgebra, int spacedim>
  Problem<dim, LinearAlgebra, spacedim>::Problem(const Parameter &prm)
    : ProblemBase<dim, LinearAlgebra, spacedim>(prm)
  {
    TimerOutput::Scope t(getTimer(), "initialize_problem");

    // prepare operator
    poisson_operator = create_operator<dim, LinearAlgebra, spacedim>(prm.operator_type,
                                                                     this->mapping_collection,
                                                                     this->quadrature_collection,
                                                                     this->fe_collection);

    // choose functions
    if (prm.grid_type == "reentrant corner")
      {
        boundary_function = Factory::create_function<dim>("reentrant corner");
        this->solution_function = Factory::create_function<dim>("reentrant corner");
        // rhs_function      = Factory::create_function<dim>("zero");
      }
    else
      {
        Assert(false, ExcNotImplemented());
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

    SolverControl solver_control(system_rhs.size(), 1e-12 * system_rhs.l2_norm());

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
            solve_gmg<dim, LinearAlgebra, spacedim>(solver_control,
                                                    *poisson_operator,
                                                    completely_distributed_solution,
                                                    system_rhs,
                                                    /*boundary_values=*/mapping_collection,
                                                    dof_handler);
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

          if (prm.operator_type == "MatrixBased")
            Log::log_nonzero_elements(poisson_operator->get_system_matrix());

          solve();

          compute_errors();
          adaptation_strategy->estimate_mark();

          if ((prm.output_frequency > 0) && (cycle % prm.output_frequency == 0))
            output_results();
        }

        Log::log_timings();

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
