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

#ifndef poisson_solvers_h
#define poisson_solvers_h


#include <deal.II/matrix_free/tools.h>

#include <multigrid/asm.h>
#include <multigrid/diagonal_matrix_timer.h>
#include <multigrid/extended_diagonal.h>
#include <multigrid/mg_solver.h>
#include <multigrid/parameter.h>
#include <multigrid/patch_indices.h>
#include <multigrid/reduce_and_assemble.h>


namespace Poisson
{
  template <int dim, typename LinearAlgebra, int spacedim>
  static void
  solve_amg(dealii::SolverControl                            &solver_control,
            const OperatorType<dim, LinearAlgebra, spacedim> &poisson_operator,
            typename LinearAlgebra::Vector                   &dst,
            const typename LinearAlgebra::Vector             &src)
  {
    typename LinearAlgebra::PreconditionAMG::AdditionalData data;
    if constexpr (std::is_same_v<LinearAlgebra, PETSc>)
      {
        data.symmetric_operator = true;
      }
    else if constexpr (std::is_same_v<LinearAlgebra, Trilinos> ||
                       std::is_same_v<LinearAlgebra, dealiiTrilinos>)
      {
        data.elliptic              = true;
        data.higher_order_elements = true;
      }
    else
      {
        Assert(false, dealii::ExcNotImplemented());
      }

    typename LinearAlgebra::PreconditionAMG preconditioner;
    preconditioner.initialize(poisson_operator.get_system_matrix(), data);

    typename LinearAlgebra::SolverCG cg(solver_control);

    if constexpr (std::is_same_v<LinearAlgebra, dealiiTrilinos>)
      {
        cg.solve(poisson_operator, dst, src, preconditioner);
      }
    else
      {
        cg.solve(poisson_operator.get_system_matrix(), dst, src, preconditioner);
      }
  }



  template <typename SmootherPreconditionerType, int dim, typename LinearAlgebra, int spacedim>
  static void
  solve_gmg(dealii::SolverControl                              &solver_control,
            const OperatorType<dim, LinearAlgebra, spacedim>   &poisson_operator,
            typename LinearAlgebra::Vector                     &dst,
            const typename LinearAlgebra::Vector               &src,
            const MGSolverParameters                           &mg_data,
            const dealii::hp::MappingCollection<dim, spacedim> &mapping_collection,
            const dealii::hp::QCollection<dim>                 &q_collection,
            const dealii::DoFHandler<dim, spacedim>            &dof_handler,
            const std::string                                  &filename_mg_level)
  {
    using namespace dealii;

    using VectorType = typename LinearAlgebra::Vector;

    TimerOutput::Scope t_mg_setup_levels(getTimer(), "mg_setup_levels");

    // Create a DoFHandler and operator for each multigrid level defined
    // by p-coarsening, as well as, create transfer operators.
    MGLevelObject<DoFHandler<dim, spacedim>>                                   dof_handlers;
    MGLevelObject<std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>> operators;

    std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>> coarse_grid_triangulations;
    if (mg_data.transfer.perform_h_transfer)
      coarse_grid_triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          dof_handler.get_triangulation());
    else
      coarse_grid_triangulations.emplace_back(
        const_cast<Triangulation<dim, spacedim> *>(&(dof_handler.get_triangulation())), [](auto &) {
          // empty deleter, since fine_triangulation_in is an external field
          // and its destructor is called somewhere else
        });

    const unsigned int n_h_levels = coarse_grid_triangulations.size() - 1;

    // Determine the number of levels.
    const auto get_max_active_fe_degree = [&](const auto &dof_handler) {
      unsigned int max = 0;

      for (auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          max = std::max(max, dof_handler.get_fe(cell->active_fe_index()).degree);

      return Utilities::MPI::max(max, MPI_COMM_WORLD);
    };

    const unsigned int n_p_levels =
      MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
        get_max_active_fe_degree(dof_handler), mg_data.transfer.p_sequence)
        .size();

    std::map<unsigned int, unsigned int> fe_index_for_degree;
    for (unsigned int i = 0; i < dof_handler.get_fe_collection().size(); ++i)
      {
        const unsigned int degree = dof_handler.get_fe(i).degree;
        Assert(fe_index_for_degree.find(degree) == fe_index_for_degree.end(),
               ExcMessage("FECollection does not contain unique degrees."));
        fe_index_for_degree[degree] = i;
      }

    unsigned int minlevel   = 0;
    unsigned int minlevel_p = n_h_levels;
    unsigned int maxlevel   = n_h_levels + n_p_levels - 1;

    // Allocate memory for all levels.
    dof_handlers.resize(minlevel, maxlevel);
    operators.resize(minlevel, maxlevel);

    // Loop from max to min level and set up DoFHandler with coarser mesh...
    for (unsigned int l = 0; l < n_h_levels; ++l)
      {
        dof_handlers[l].reinit(*coarse_grid_triangulations[l]);
        dof_handlers[l].distribute_dofs(dof_handler.get_fe_collection());
      }

    // ... with lower polynomial degrees
    for (unsigned int i = 0, l = maxlevel; i < n_p_levels; ++i, --l)
      {
        dof_handlers[l].reinit(dof_handler.get_triangulation());

        if (l == maxlevel) // finest level
          {
            auto &dof_handler_mg = dof_handlers[l];

            auto cell_other = dof_handler.begin_active();
            for (auto &cell : dof_handler_mg.active_cell_iterators())
              {
                if (cell->is_locally_owned())
                  cell->set_active_fe_index(cell_other->active_fe_index());
                cell_other++;
              }
          }
        else // coarse level
          {
            auto &dof_handler_fine   = dof_handlers[l + 1];
            auto &dof_handler_coarse = dof_handlers[l + 0];

            auto cell_other = dof_handler_fine.begin_active();
            for (auto &cell : dof_handler_coarse.active_cell_iterators())
              {
                if (cell->is_locally_owned())
                  {
                    const unsigned int next_degree =
                      MGTransferGlobalCoarseningTools::create_next_polynomial_coarsening_degree(
                        cell_other->get_fe().degree, mg_data.transfer.p_sequence);
                    Assert(fe_index_for_degree.find(next_degree) != fe_index_for_degree.end(),
                           ExcMessage("Next polynomial degree in sequence "
                                      "does not exist in FECollection."));

                    cell->set_active_fe_index(fe_index_for_degree[next_degree]);
                  }
                cell_other++;
              }
          }

        dof_handlers[l].distribute_dofs(dof_handler.get_fe_collection());
      }

    // Create data structures on each multigrid level.
    MGLevelObject<AffineConstraints<typename VectorType::value_type>> constraints(minlevel,
                                                                                  maxlevel);

    MGLevelObject<std::shared_ptr<SmootherPreconditionerType>> smoother_preconditioners(minlevel,
                                                                                        maxlevel);

    //
    // TODO: Generalise, maybe for operator and blockoperatorbase?
    //       Pass this part as lambda function?
    //       Or just pass vector target?
    //
    for (unsigned int level = minlevel; level <= maxlevel; level++)
      {
        const auto &dof_handler = dof_handlers[level];
        auto       &constraint  = constraints[level];

        Partitioning partitioning;
        partitioning.reinit(dof_handler);

        // ... constraints (with homogenous Dirichlet BC)
        constraint.reinit(partitioning.get_relevant_dofs());

        DoFTools::make_hanging_node_constraints(dof_handler, constraint);
        VectorTools::interpolate_boundary_values(
          mapping_collection, dof_handler, 0, Functions::ZeroFunction<dim>(), constraint);
        constraint.close();

        // ... operator (just like on the finest level)
        operators[level] = poisson_operator.replicate();
        operators[level]->reinit(partitioning, dof_handler, constraint);

        // TODO: Also store sparsity patterns?


        // WIP: build smoother preconditioners here
        // necessary on all levels or just minlevel+1 to maxlevel?

        if constexpr (std::is_same_v<SmootherPreconditionerType, DiagonalMatrixTimer<VectorType>>)
          {
            smoother_preconditioners[level] = std::make_shared<SmootherPreconditionerType>();
            operators[level]->compute_inverse_diagonal(
              smoother_preconditioners[level]->get_vector());
          }
        else if constexpr (std::is_same_v<SmootherPreconditionerType, PreconditionASM<VectorType>>)
          {
            const auto patch_indices = prepare_patch_indices(dof_handler, constraint);

            if (level == maxlevel)
              Log::log_patch_dofs(patch_indices, dof_handler);

            // full matrix
            // TODO: this is a nasty way to get the sparsity pattern
            // so far I only created temporary sparsity patterns in the LinearAlgebra namespace,
            // but they are no longer available here
            // so for the sake of trying ASM out, I'll just create another one here
            const unsigned int myid =
              dealii::Utilities::MPI::this_mpi_process(dof_handler.get_communicator());
            typename LinearAlgebra::SparsityPattern sparsity_pattern;
            sparsity_pattern.reinit(partitioning.get_owned_dofs(),
                                    partitioning.get_owned_dofs(),
                                    partitioning.get_relevant_dofs(),
                                    dof_handler.get_communicator());
            DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, constraint, false, myid);
            sparsity_pattern.compress();

            smoother_preconditioners[level] =
              std::make_shared<SmootherPreconditionerType>(std::move(patch_indices));
            smoother_preconditioners[level]->initialize(operators[level]->get_system_matrix(),
                                                        sparsity_pattern,
                                                        dof_handler);
          }
        else if constexpr (std::is_same_v<SmootherPreconditionerType,
                                          PreconditionExtendedDiagonal<VectorType>>)
          {
            const auto patch_indices = prepare_patch_indices(dof_handler, constraint);

            if (level == maxlevel)
              Log::log_patch_dofs(patch_indices, dof_handler);

            // full matrix
            // const unsigned int myid = dealii::Utilities::MPI::this_mpi_process(communicator);
            // DynamicSparsityPattern dsp(relevant_dofs);
            // DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false, myid);
            // SparsityTools::distribute_sparsity_pattern(dsp, owned_dofs, communicator,
            // relevant_dofs);

            // reduced matrix
            AffineConstraints<double> constraints_reduced;
            constraints_reduced.reinit(partitioning.get_owned_dofs(),
                                       partitioning.get_relevant_dofs());

            const auto all_indices_relevant =
              extract_relevant(patch_indices, partitioning, dof_handler);

            std::set<types::global_dof_index> all_indices_assemble;
            reduce_constraints(constraint,
                               DoFTools::extract_locally_active_dofs(dof_handler),
                               all_indices_relevant,
                               constraints_reduced,
                               all_indices_assemble);

            // TODO: only works for Trilinos so far
            typename LinearAlgebra::SparsityPattern reduced_sparsity_pattern;
            reduced_sparsity_pattern.reinit(partitioning.get_owned_dofs(),
                                            partitioning.get_owned_dofs(),
                                            partitioning.get_relevant_dofs(),
                                            dof_handler.get_communicator());
            make_sparsity_pattern(dof_handler,
                                  all_indices_assemble,
                                  reduced_sparsity_pattern,
                                  constraints_reduced);
            reduced_sparsity_pattern.compress();

            typename LinearAlgebra::SparseMatrix reduced_sparse_matrix;
            reduced_sparse_matrix.reinit(reduced_sparsity_pattern);
            partially_assemble_poisson(dof_handler,
                                       constraints_reduced,
                                       q_collection,
                                       all_indices_assemble,
                                       reduced_sparse_matrix);

            VectorType inverse_diagonal;
            operators[level]->compute_inverse_diagonal(inverse_diagonal);

            smoother_preconditioners[level] =
              std::make_shared<SmootherPreconditionerType>(std::move(patch_indices));
            // smoother_preconditioners[level]->->initialize(operators[level]->get_system_matrix(),
            //                                               dsp,
            //                                               inverse_diagonal,
            //                                               all_indices_relevant);
            smoother_preconditioners[level]->initialize(reduced_sparse_matrix,
                                                        reduced_sparsity_pattern,
                                                        inverse_diagonal,
                                                        all_indices_relevant);
          }
        else
          {
            AssertThrow(false, ExcNotImplemented());
          }
      }

    t_mg_setup_levels.stop();



    TimerOutput::Scope t_mg_reinit_transfer(getTimer(), "mg_reinit_transfer");

    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;
    transfers.resize(minlevel, maxlevel);

    // Set up intergrid operators.
    for (unsigned int level = minlevel; level < minlevel_p; ++level)
      transfers[level + 1].reinit_geometric_transfer(dof_handlers[level + 1],
                                                     dof_handlers[level],
                                                     constraints[level + 1],
                                                     constraints[level]);

    for (unsigned int level = minlevel_p; level < maxlevel; ++level)
      transfers[level + 1].reinit_polynomial_transfer(dof_handlers[level + 1],
                                                      dof_handlers[level],
                                                      constraints[level + 1],
                                                      constraints[level]);

    // Collect transfer operators within a single operator as needed by
    // the Multigrid solver class.
    MGTransferGlobalCoarsening<dim, VectorType> transfer(transfers, [&](const auto l, auto &vec) {
      operators[l]->initialize_dof_vector(vec);
    });

    t_mg_reinit_transfer.stop();



    TimerOutput::Scope t_mg_solve(getTimer(), "mg_solve");

    // Proceed to solve the problem with multigrid.
    mg_solve(solver_control,
             dst,
             src,
             mg_data,
             dof_handler,
             poisson_operator,
             operators,
             smoother_preconditioners,
             transfer,
             filename_mg_level);

    t_mg_solve.stop();
  }
} // namespace Poisson


#endif
