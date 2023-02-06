// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the deal.II authors
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

#ifndef stokes_matrixfree_gmg_h
#define stokes_matrixfree_gmg_h


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <base/linear_algebra.h>
#include <stokes/matrixfree/block_schur_preconditioner.h>
#include <stokes/matrixfree/a_block_operator.h>
#include <stokes/matrixfree/schur_block_operator.h>
#include <stokes/matrixfree/stokes_operator.h>

#include <mg_solver.h>


namespace StokesMatrixFree
{
  template <int dim, typename LinearAlgebra, int spacedim = dim>
  static void
  solve_gmg(dealii::SolverControl                                        &solver_control_refined,
            const StokesMatrixFree::StokesOperator<dim, LinearAlgebra, spacedim> &stokes_operator,
            const OperatorType<dim, LinearAlgebra, spacedim>             &a_block_operator,
            const OperatorType<dim, LinearAlgebra, spacedim>             &schur_block_operator,
            typename LinearAlgebra::BlockVector                          &dst,
            const typename LinearAlgebra::BlockVector                    &src,
            const dealii::hp::MappingCollection<dim, spacedim>           &mapping_collection,
            const std::vector<const dealii::DoFHandler<dim, spacedim> *> &stokes_dof_handlers)
  {
    // poisson has mappingcollection and dofhandler as additional parameters

    using namespace dealii;

    using VectorType = typename LinearAlgebra::Vector;

    const MGSolverParameters mg_data;

    // TODO: this is only temporary
    // only work on velocity dofhandlers for now
    const DoFHandler<dim, spacedim> &dof_handler = *(stokes_dof_handlers[0]);

    // Create a DoFHandler and operator for each multigrid level defined
    // by p-coarsening, as well as, create transfer operators.
    MGLevelObject<DoFHandler<dim, spacedim>>                                   dof_handlers;
    MGLevelObject<std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>> operators;
    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>>                         transfers;

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
    transfers.resize(minlevel, maxlevel);

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
        // TODO: externalize this
        VectorTools::interpolate_boundary_values(
          mapping_collection, dof_handler, 0, Functions::ZeroFunction<dim>(dim), constraint);

        constraint.close();

        // ... operator (just like on the finest level)
        operators[level] = a_block_operator.replicate();
        operators[level]->reinit(partitioning, dof_handler, constraint);
      }

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

    //
    // setup coarse solver
    //

    //using LevelMatrixType = StokesMatrixFree::ABlockOperator<dim, LinearAlgebra, spacedim>;
    using LevelMatrixType = OperatorType<dim, LinearAlgebra, spacedim>;
    using MGTransferType = MGTransferGlobalCoarsening<dim, VectorType>;

    using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
    using SmootherType =
      PreconditionChebyshev<LevelMatrixType, VectorType, SmootherPreconditionerType>;
    using PreconditionerType = PreconditionMG<dim, VectorType, MGTransferType>;

    const unsigned int min_level = operators.min_level();
    const unsigned int max_level = operators.max_level();

    // Initialize level operators.
    mg::Matrix<VectorType> mg_matrix(operators); // operators = mg_matrices in mg_solver.h

    // Initialize smoothers.
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level, max_level);

    for (unsigned int level = min_level; level <= max_level; level++)
      {
        smoother_data[level].preconditioner = std::make_shared<SmootherPreconditionerType>();
        operators[level]->compute_inverse_diagonal(
          smoother_data[level].preconditioner->get_vector());
        smoother_data[level].smoothing_range     = mg_data.smoother.smoothing_range;
        smoother_data[level].degree              = mg_data.smoother.degree;
        smoother_data[level].eig_cg_n_iterations = mg_data.smoother.eig_cg_n_iterations;
      }

    MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;
    mg_smoother.initialize(operators, smoother_data);

    // Initialize coarse-grid solver.
    ReductionControl coarse_grid_solver_control(mg_data.coarse_solver.maxiter,
                                                mg_data.coarse_solver.abstol,
                                                mg_data.coarse_solver.reltol,
                                                false,
                                                false);
    SolverCG<VectorType> coarse_grid_solver(coarse_grid_solver_control);

    std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;
#ifdef DEAL_II_WITH_TRILINOS
    TrilinosWrappers::PreconditionAMG precondition_amg;
    TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
    amg_data.smoother_sweeps = mg_data.coarse_solver.smoother_sweeps;
    amg_data.n_cycles        = mg_data.coarse_solver.n_cycles;
    amg_data.smoother_type   = mg_data.coarse_solver.smoother_type.c_str();

    // CG with AMG as preconditioner
    precondition_amg.initialize(operators[min_level]->get_system_matrix(), amg_data);

    mg_coarse = std::make_unique<MGCoarseGridIterativeSolver<VectorType,
                                                             SolverCG<VectorType>,
                                                             LevelMatrixType,
                                                             decltype(precondition_amg)>>(
      coarse_grid_solver, *operators[min_level], precondition_amg);
#endif

    // Create multigrid object.
    Multigrid<VectorType> mg_a_block(mg_matrix, *mg_coarse, transfer, mg_smoother, mg_smoother);

    // Convert it to a preconditioner.
    PreconditionerType a_block_preconditioner(dof_handler, mg_a_block, transfer);

    // TODO: use something more/less complex?
    // PreconditionIdentity precon_schur;
    typename LinearAlgebra::PreconditionJacobi schur_block_preconditioner;
    schur_block_preconditioner.initialize(schur_block_operator.get_system_matrix());

    const LinearSolversMatrixFree::BlockSchurPreconditioner<
      LinearAlgebra,
      StokesMatrixFree::StokesOperator<dim, LinearAlgebra, spacedim>,
      OperatorType<dim, LinearAlgebra, spacedim>,
      OperatorType<dim, LinearAlgebra, spacedim>,
      PreconditionerType,
      typename LinearAlgebra::PreconditionJacobi>
      preconditioner(stokes_operator,
                     a_block_operator,
                     schur_block_operator,
                     a_block_preconditioner,
                     schur_block_preconditioner,
                     true);

    // set up solver
    dealii::PrimitiveVectorMemory<typename LinearAlgebra::BlockVector> mem;

    typename dealii::SolverFGMRES<typename LinearAlgebra::BlockVector>::AdditionalData fgmres_data(
      50);
    dealii::SolverFGMRES<typename LinearAlgebra::BlockVector> solver(solver_control_refined,
                                                                     mem,
                                                                     fgmres_data);

    solver.solve(stokes_operator, dst, src, preconditioner);
  }
} // namespace StokesMatrixFree


#endif
