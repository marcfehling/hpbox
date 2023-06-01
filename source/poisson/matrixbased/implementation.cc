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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <base/global.h>
#include <base/linear_algebra.h>
#include <base/log.h>
#include <factory.h>
#include <poisson/matrixbased/implementation.h>
#include <poisson/matrixbased/amg.h>
//#include <poisson/gmg.h>
#include <poisson/matrixbased/poisson_operator.h>
//#include <poisson/matrixfree/poisson_operator.h>
#include <problem.h>

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
            // return std::make_unique<
            //   PoissonMatrixFree::PoissonOperator<dim, LinearAlgebra, spacedim>>(
            //   std::forward<Args>(args)...);
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
  namespace MatrixBased
  {
  template <int dim, typename LinearAlgebra, int spacedim>
  Implementation<dim, LinearAlgebra, spacedim>::Implementation(Problem<dim, LinearAlgebra, spacedim> &problem)
    : problem(problem)
  {}



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Implementation<dim, LinearAlgebra, spacedim>::reinit()
  {
    TimerOutput::Scope t(getTimer(), "initialize_problem");

    // prepare collections
    problem.mapping_collection.push_back(MappingQ1<dim, spacedim>());

     for (unsigned int degree = 1; degree <= problem.prm.prm_adaptation.max_p_degree; ++degree)
       {
         problem.fe_collection.push_back(FE_Q<dim, spacedim>(degree));
         problem.quadrature_collection.push_back(QGauss<dim>(degree + 1));
       }

    // prepare operator
    poisson_operator = create_operator<dim, LinearAlgebra, spacedim>(problem.prm.operator_type,
                                        problem.mapping_collection,
                                        problem.quadrature_collection,
                                        problem.fe_collection);

    // choose functions
    // TODO: Move this to general Problem
    if (problem.prm.grid_type == "reentrant corner")
      {
        problem.boundary_function = Factory::create_function<dim>("reentrant corner");
        problem.solution_function = Factory::create_function<dim>("reentrant corner");
        // rhs_function      = Factory::create_function<dim>("zero");
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Implementation<dim, LinearAlgebra, spacedim>::setup_system()
  {
    TimerOutput::Scope t(getTimer(), "setup_system");

    Partitioning partitioning;

    {
      TimerOutput::Scope t(getTimer(), "distribute_dofs");

      problem.dof_handler.distribute_dofs(problem.fe_collection);
      partitioning.reinit(problem.dof_handler);
    }

    {
      TimerOutput::Scope t(getTimer(), "reinit_vectors");

      problem.locally_relevant_solution.reinit(partitioning.get_owned_dofs(),
                                             partitioning.get_relevant_dofs(),
                                             problem.mpi_communicator);
      // TODO: partitioning get communicator?
    }

    {
      TimerOutput::Scope t(getTimer(), "make_constraints");

      problem.constraints.clear();
      problem.constraints.reinit(partitioning.get_relevant_dofs());

      DoFTools::make_hanging_node_constraints(problem.dof_handler, problem.constraints);

      VectorTools::interpolate_boundary_values(
        problem.mapping_collection, problem.dof_handler, 0, *(problem.boundary_function), problem.constraints);

#ifdef DEBUG
      // We have not dealt with chains of constraints on ghost cells yet.
      // Thus, we are content with verifying their consistency for now.
      const std::vector<IndexSet> locally_owned_dofs_per_processor =
        Utilities::MPI::all_gather(problem.mpi_communicator, problem.dof_handler.locally_owned_dofs());

      IndexSet locally_active_dofs;
      DoFTools::extract_locally_active_dofs(problem.dof_handler, locally_active_dofs);

      AssertThrow(problem.constraints.is_consistent_in_parallel(locally_owned_dofs_per_processor,
                                                        locally_active_dofs,
                                                        problem.mpi_communicator,
                                                        /*verbose=*/true),
                  ExcMessage("AffineConstraints object contains inconsistencies!"));
#endif
      problem.constraints.close();
    }

    {
      TimerOutput::Scope t(getTimer(), "reinit_operator");

      poisson_operator->reinit(partitioning, problem.dof_handler, problem.constraints, problem.system_rhs, nullptr);
    }

    Log::log_hp_diagnostics(problem.triangulation, problem.dof_handler, problem.constraints);

    // TODO: if not matrixfree, or ask for parameter set
    Log::log_nonzero_elements(poisson_operator->get_system_matrix());
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Implementation<dim, LinearAlgebra, spacedim>::solve()
  {
    TimerOutput::Scope t(getTimer(), "solve");

    // We need to introduce a vector that does not contain all ghost elements.
    typename LinearAlgebra::Vector completely_distributed_solution;
    poisson_operator->initialize_dof_vector(completely_distributed_solution);

    SolverControl solver_control(problem.system_rhs.size(), 1e-12 * problem.system_rhs.l2_norm());

    Assert(problem.prm.solver_type == "AMG", ExcNotImplemented());
//    if (problem.prm.solver_type == "AMG")
//      {
        solve_amg<dim, LinearAlgebra, spacedim>(solver_control,
                                                *poisson_operator,
                                                completely_distributed_solution,
                                                problem.system_rhs);
//      }
//    else if (problem.prm.solver_type == "GMG")
//      {
//        if constexpr (std::is_same_v<LinearAlgebra, dealiiTrilinos>)
//          {
//            solve_gmg<dim, LinearAlgebra, spacedim>(solver_control,
//                                                    *poisson_operator,
//                                                    completely_distributed_solution,
//                                                    problem.system_rhs,
//                                                    /*boundary_values=*/problem.mapping_collection,
//                                                    problem.dof_handler);
//          }
//        else
//          {
//            AssertThrow(false, ExcMessage("GMG is only available with dealii & Trilinos!"));
//          }
//      }
//    else
//      {
//        Assert(false, ExcNotImplemented());
//      }

    Log::log_iterations(solver_control);

    problem.constraints.distribute(completely_distributed_solution);

    problem.locally_relevant_solution = completely_distributed_solution;
    problem.locally_relevant_solution.update_ghost_values();
  }



  // explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class Implementation<2, dealiiTrilinos, 2>;
  template class Implementation<3, dealiiTrilinos, 3>;
  template class Implementation<2, Trilinos, 2>;
  template class Implementation<3, Trilinos, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
  template class Implementation<2, PETSc, 2>;
  template class Implementation<3, PETSc, 3>;
#endif
  } // namespace MatrixBased
} // namespace Poisson
