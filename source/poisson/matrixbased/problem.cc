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
#include <poisson/matrixbased/amg.h>
//#include <poisson/gmg.h>
#include <poisson/matrixbased/poisson_operator.h>
//#include <poisson/matrixfree/poisson_operator.h>
#include <poisson/matrixbased/problem.h>

#include <ctime>
#include <iomanip>
#include <sstream>

using namespace dealii;


namespace Poisson
{
  namespace MatrixBased
  {
  template <int dim, typename LinearAlgebra, int spacedim>
  Problem<dim, LinearAlgebra, spacedim>::Problem(const Parameter &prm)
    : ProblemBase<dim, LinearAlgebra, spacedim>(prm)
    , poisson_operator(this->mapping_collection, this->quadrature_collection, this->fe_collection)
  {
    TimerOutput::Scope t(getTimer(), "initialize_problem");

    // prepare collections
    this->mapping_collection.push_back(MappingQ1<dim, spacedim>());

     for (unsigned int degree = 1; degree <= prm.prm_adaptation.max_p_degree; ++degree)
       {
         this->fe_collection.push_back(FE_Q<dim, spacedim>(degree));
         this->quadrature_collection.push_back(QGauss<dim>(degree + 1));
       }

     // choose adaptation strategy
       this->adaptation_strategy = Factory::create_adaptation<dim, typename LinearAlgebra::Vector, spacedim>(
           this->prm.adaptation_type,
           this->prm.prm_adaptation,
           this->locally_relevant_solution,
           this->fe_collection,
           this->dof_handler,
           this->triangulation);

    // choose functions
    if (this->prm.grid_type == "reentrant corner")
      {
        this->boundary_function = Factory::create_function<dim>("reentrant corner");
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

      this->dof_handler.distribute_dofs(this->fe_collection);
      this->partitioning.reinit(this->dof_handler);
    }

    {
      TimerOutput::Scope t(getTimer(), "reinit_vectors");

      this->locally_relevant_solution.reinit(this->partitioning.get_owned_dofs(),
                                             this->partitioning.get_relevant_dofs(),
                                             this->mpi_communicator);
    }

    {
      TimerOutput::Scope t(getTimer(), "make_constraints");

      this->constraints.clear();
      this->constraints.reinit(this->partitioning.get_relevant_dofs());

      DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);

      VectorTools::interpolate_boundary_values(
        this->mapping_collection, this->dof_handler, 0, *(this->boundary_function), this->constraints);

#ifdef DEBUG
      // We have not dealt with chains of constraints on ghost cells yet.
      // Thus, we are content with verifying their consistency for now.
      const std::vector<IndexSet> locally_owned_dofs_per_processor =
        Utilities::MPI::all_gather(this->mpi_communicator, this->dof_handler.locally_owned_dofs());

      IndexSet locally_active_dofs;
      DoFTools::extract_locally_active_dofs(this->dof_handler, locally_active_dofs);

      AssertThrow(constraints.is_consistent_in_parallel(locally_owned_dofs_per_processor,
                                                        locally_active_dofs,
                                                        this->mpi_communicator,
                                                        /*verbose=*/true),
                  ExcMessage("AffineConstraints object contains inconsistencies!"));
#endif
      this->constraints.close();
    }

    {
      TimerOutput::Scope t(getTimer(), "reinit_operator");

      poisson_operator.reinit(this->partitioning, this->dof_handler, this->constraints, this->system_rhs, nullptr);
    }

    Log::log_hp_diagnostics(this->triangulation, this->dof_handler, this->constraints);
    Log::log_nonzero_elements(poisson_operator.get_system_matrix());
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::solve()
  {
    TimerOutput::Scope t(getTimer(), "solve");

    // We need to introduce a vector that does not contain all ghost elements.
    typename LinearAlgebra::Vector completely_distributed_solution;
    poisson_operator.initialize_dof_vector(completely_distributed_solution);

    SolverControl solver_control(this->system_rhs.size(), 1e-12 * this->system_rhs.l2_norm());

    Assert(this->prm.solver_type == "AMG", ExcNotImplemented());
//    if (this->prm.solver_type == "AMG")
//      {
        solve_amg<dim, LinearAlgebra, spacedim>(solver_control,
                                                poisson_operator,
                                                completely_distributed_solution,
                                                this->system_rhs);
//      }
//    else if (this->prm.solver_type == "GMG")
//      {
//        if constexpr (std::is_same_v<LinearAlgebra, dealiiTrilinos>)
//          {
//            solve_gmg<dim, LinearAlgebra, spacedim>(solver_control,
//                                                    poisson_operator,
//                                                    completely_distributed_solution,
//                                                    this->system_rhs,
//                                                    /*boundary_values=*/this->mapping_collection,
//                                                    this->dof_handler);
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

    this->constraints.distribute(completely_distributed_solution);

    this->locally_relevant_solution = completely_distributed_solution;
    this->locally_relevant_solution.update_ghost_values();
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
  }
} // namespace Poisson
