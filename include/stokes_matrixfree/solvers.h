// ---------------------------------------------------------------------
//
// Copyright (C) 2022 - 2023 by the deal.II authors
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

#ifndef stokes_matrixfree_solvers_h
#define stokes_matrixfree_solvers_h


#include <deal.II/base/convergence_table.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
// #include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <linear_algebra.h>
#include <multigrid/asm.h>
#include <multigrid/diagonal_matrix_timer.h>
#include <multigrid/extended_diagonal.h>
#include <multigrid/mg_solver.h>
#include <multigrid/parameter.h>
#include <multigrid/patch_indices.h>
#include <multigrid/reduce_and_assemble.h>
#include <stokes_matrixfree/operators.h>


namespace StokesMatrixFree
{
  template <typename LinearAlgebra,
            typename StokesMatrixType,
            typename ABlockMatrixType,
            typename SchurComplementMatrixType,
            typename ABlockPreconditionerType,
            typename SchurComplementPreconditionerType>
  class BlockSchurPreconditioner : public dealii::Subscriptor
  {
  public:
    BlockSchurPreconditioner(
      const StokesMatrixType                  &stokes_matrix,
      const ABlockMatrixType                  &a_block,
      const SchurComplementMatrixType         &schur_complement_block,
      const ABlockPreconditionerType          &a_block_preconditioner,
      const SchurComplementPreconditionerType &schur_complement_preconditioner,
      const bool                               do_solve_A,
      const bool                               do_solve_Schur_complement)
      : stokes_matrix(&stokes_matrix)
      , a_block(&a_block)
      , schur_complement_block(&schur_complement_block)
      , a_block_preconditioner(a_block_preconditioner)
      , schur_complement_preconditioner(schur_complement_preconditioner)
      , do_solve_A(do_solve_A)
      , do_solve_Schur_complement(do_solve_Schur_complement)
      , max_A_iterations(0)
      , max_Schur_iterations(0)
    {}

    ~BlockSchurPreconditioner()
    {
      if (max_A_iterations > 0)
        {
          getPCOut() << "   A solved in max. " << max_A_iterations << " iterations." << std::endl;
          getTable().add_value("max_a_iterations", max_A_iterations);
        }

      if (max_Schur_iterations > 0)
        {
          getPCOut() << "   Schur complement solved in max. " << max_Schur_iterations
                     << " iterations." << std::endl;
          getTable().add_value("max_schur_iterations", max_Schur_iterations);
        }
    }

    void
    vmult(typename LinearAlgebra::BlockVector       &dst,
          const typename LinearAlgebra::BlockVector &src) const
    {
      dealii::TimerOutput::Scope t(getTimer(), "vmult_BlockSchurPreconditioner");

      // This needs to be done explicitly, as GMRES does not initialize the data of the vector dst
      // before calling us. Otherwise we might use random data as our initial guess.
      // See also: https://github.com/geodynamics/aspect/pull/4973
      dst = 0.;

      if (do_solve_Schur_complement)
        {
          dealii::ReductionControl solver_control(5000, 1e-6 * src.block(1).l2_norm(), 1e-2);
          typename LinearAlgebra::SolverCG solver(solver_control);

          solver.solve(*schur_complement_block,
                       dst.block(1),
                       src.block(1),
                       schur_complement_preconditioner);

          max_Schur_iterations = std::max(solver_control.last_step(), max_Schur_iterations);
        }
      else
        schur_complement_preconditioner.vmult(dst.block(1), src.block(1));

      dst.block(1) *= -1.0;

      typename LinearAlgebra::BlockVector utmp;
      utmp.reinit(src);

      {
        stokes_matrix->vmult(utmp, dst); // B^T
        utmp.block(0) *= -1.0;
        utmp.block(0) += src.block(0);
      }

      if (do_solve_A == true)
        {
          dealii::ReductionControl solver_control(5000, 1e-2 * utmp.block(0).l2_norm(), 1e-2);
          typename LinearAlgebra::SolverCG solver(solver_control);

          solver.solve(*a_block, dst.block(0), utmp.block(0), a_block_preconditioner);

          max_A_iterations = std::max(solver_control.last_step(), max_A_iterations);
        }
      else
        a_block_preconditioner.vmult(dst.block(0), utmp.block(0));
    }

  private:
    const dealii::SmartPointer<const StokesMatrixType>          stokes_matrix;
    const dealii::SmartPointer<const ABlockMatrixType>          a_block;
    const dealii::SmartPointer<const SchurComplementMatrixType> schur_complement_block;

    const ABlockPreconditionerType          &a_block_preconditioner;
    const SchurComplementPreconditionerType &schur_complement_preconditioner;

    const bool do_solve_A;
    const bool do_solve_Schur_complement;

    mutable unsigned int max_A_iterations;
    mutable unsigned int max_Schur_iterations;
  };



  template <int dim, typename LinearAlgebra, int spacedim = dim>
  static void
  solve_amg(dealii::SolverControl &solver_control_refined,
            const StokesMatrixFree::StokesOperator<dim, LinearAlgebra, spacedim> &stokes_operator,
            const OperatorType<dim, LinearAlgebra, spacedim>                     &a_block_operator,
            const OperatorType<dim, LinearAlgebra, spacedim> &schur_block_operator,
            typename LinearAlgebra::BlockVector              &dst,
            const typename LinearAlgebra::BlockVector        &src)
  {
    typename LinearAlgebra::PreconditionAMG::AdditionalData Amg_data;
    if constexpr (std::is_same<LinearAlgebra, PETSc>::value)
      {
        Amg_data.symmetric_operator = true;
        Amg_data.n_sweeps_coarse    = 2;
        Amg_data.strong_threshold   = 0.02;
      }
    else if constexpr (std::is_same<LinearAlgebra, Trilinos>::value ||
                       std::is_same<LinearAlgebra, dealiiTrilinos>::value)
      {
        Amg_data.elliptic              = true;
        Amg_data.higher_order_elements = true;
        Amg_data.smoother_sweeps       = 2;
        Amg_data.aggregation_threshold = 0.02;
      }
    else
      {
        Assert(false, dealii::ExcNotImplemented());
      }

    typename LinearAlgebra::PreconditionJacobi Mp_preconditioner;
    typename LinearAlgebra::PreconditionAMG    Amg_preconditioner;

    Mp_preconditioner.initialize(schur_block_operator.get_system_matrix());
    Amg_preconditioner.initialize(a_block_operator.get_system_matrix(), Amg_data);

    //
    // TODO: System Matrix or operator? See below
    //
    const BlockSchurPreconditioner<LinearAlgebra,
                                   StokesMatrixFree::StokesOperator<dim, LinearAlgebra, spacedim>,
                                   typename LinearAlgebra::SparseMatrix,
                                   typename LinearAlgebra::SparseMatrix,
                                   typename LinearAlgebra::PreconditionAMG,
                                   typename LinearAlgebra::PreconditionJacobi>
      preconditioner(stokes_operator,
                     a_block_operator.get_system_matrix(),
                     schur_block_operator.get_system_matrix(),
                     Amg_preconditioner,
                     Mp_preconditioner,
                     /*do_solve_A=*/false,
                     /*do_solve_Schur_complement=*/true);

    // set up solver
    dealii::PrimitiveVectorMemory<typename LinearAlgebra::BlockVector> mem;

    typename dealii::SolverFGMRES<typename LinearAlgebra::BlockVector>::AdditionalData fgmres_data(
      50);
    dealii::SolverFGMRES<typename LinearAlgebra::BlockVector> solver(solver_control_refined,
                                                                     mem,
                                                                     fgmres_data);

    solver.solve(stokes_operator, dst, src, preconditioner);
  }



  template <typename SmootherPreconditionerType, int dim, typename LinearAlgebra, int spacedim>
  static void
  solve_gmg(dealii::SolverControl &solver_control_refined,
            const StokesMatrixFree::StokesOperator<dim, LinearAlgebra, spacedim> &stokes_operator,
            const OperatorType<dim, LinearAlgebra, spacedim>                     &a_block_operator,
            const OperatorType<dim, LinearAlgebra, spacedim>             &schur_block_operator,
            typename LinearAlgebra::BlockVector                          &dst,
            const typename LinearAlgebra::BlockVector                    &src,
            const MGSolverParameters                                     &mg_data,
            const dealii::hp::MappingCollection<dim, spacedim>           &mapping_collection,
            const dealii::hp::QCollection<dim>                           &q_collection_v,
            const std::vector<const dealii::DoFHandler<dim, spacedim> *> &stokes_dof_handlers,
            const std::string                                            &filename_mg_level)
  {
    // poisson has mappingcollection and dofhandler as additional parameters

    using namespace dealii;

    using VectorType = typename LinearAlgebra::Vector;

    TimerOutput::Scope t_mg_setup_levels(getTimer(), "mg_setup_levels");

    // TODO: this is only temporary
    // only work on velocity dofhandlers for now
    const DoFHandler<dim, spacedim> &a_dof_handler = *(stokes_dof_handlers[0]);

    // Create a DoFHandler and operator for each multigrid level defined
    // by p-coarsening, as well as, create transfer operators.
    MGLevelObject<DoFHandler<dim, spacedim>>                                   a_dof_handlers;
    MGLevelObject<std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>> a_operators;

    std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>> coarse_grid_triangulations;
    if (mg_data.transfer.perform_h_transfer)
      coarse_grid_triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          a_dof_handler.get_triangulation());
    else
      coarse_grid_triangulations.emplace_back(
        const_cast<Triangulation<dim, spacedim> *>(&(a_dof_handler.get_triangulation())), [](auto &) {
          // empty deleter, since fine_triangulation_in is an external field
          // and its destructor is called somewhere else
        });

    const unsigned int n_h_levels = coarse_grid_triangulations.size() - 1;

    // Determine the number of levels.
    const auto get_max_active_fe_degree = [&](const auto &a_dof_handler) {
      unsigned int max = 0;

      for (auto &cell : a_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          max = std::max(max, a_dof_handler.get_fe(cell->active_fe_index()).degree);

      return Utilities::MPI::max(max, MPI_COMM_WORLD);
    };

    const unsigned int a_n_p_levels =
      MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
        get_max_active_fe_degree(a_dof_handler), mg_data.transfer.p_sequence)
        .size();

    std::map<unsigned int, unsigned int> a_fe_index_for_degree;
    for (unsigned int i = 0; i < a_dof_handler.get_fe_collection().size(); ++i)
      {
        const unsigned int degree = a_dof_handler.get_fe(i).degree;
        Assert(a_fe_index_for_degree.find(degree) == a_fe_index_for_degree.end(),
               ExcMessage("FECollection does not contain unique degrees."));
        a_fe_index_for_degree[degree] = i;
      }

    unsigned int a_minlevel   = 0;
    unsigned int a_minlevel_p = n_h_levels;
    unsigned int a_maxlevel   = n_h_levels + a_n_p_levels - 1;

    // Allocate memory for all levels.
    a_dof_handlers.resize(a_minlevel, a_maxlevel);
    a_operators.resize(a_minlevel, a_maxlevel);

    // Loop from max to min level and set up DoFHandler with coarser mesh...
    for (unsigned int l = 0; l < n_h_levels; ++l)
      {
        a_dof_handlers[l].reinit(*coarse_grid_triangulations[l]);
        a_dof_handlers[l].distribute_dofs(a_dof_handler.get_fe_collection());
      }

    // ... with lower polynomial degrees
    for (unsigned int i = 0, l = a_maxlevel; i < a_n_p_levels; ++i, --l)
      {
        a_dof_handlers[l].reinit(a_dof_handler.get_triangulation());

        if (l == a_maxlevel) // finest level
          {
            auto &dof_handler_mg = a_dof_handlers[l];

            auto cell_other = a_dof_handler.begin_active();
            for (auto &cell : dof_handler_mg.active_cell_iterators())
              {
                if (cell->is_locally_owned())
                  cell->set_active_fe_index(cell_other->active_fe_index());
                cell_other++;
              }
          }
        else // coarse level
          {
            auto &dof_handler_fine   = a_dof_handlers[l + 1];
            auto &dof_handler_coarse = a_dof_handlers[l + 0];

            auto cell_other = dof_handler_fine.begin_active();
            for (auto &cell : dof_handler_coarse.active_cell_iterators())
              {
                if (cell->is_locally_owned())
                  {
                    const unsigned int next_degree =
                      MGTransferGlobalCoarseningTools::create_next_polynomial_coarsening_degree(
                        cell_other->get_fe().degree, mg_data.transfer.p_sequence);
                    Assert(a_fe_index_for_degree.find(next_degree) != a_fe_index_for_degree.end(),
                           ExcMessage("Next polynomial degree in sequence "
                                      "does not exist in FECollection."));

                    cell->set_active_fe_index(a_fe_index_for_degree[next_degree]);
                  }
                cell_other++;
              }
          }

        a_dof_handlers[l].distribute_dofs(a_dof_handler.get_fe_collection());
      }

    // Create data structures on each multigrid level.
    MGLevelObject<AffineConstraints<typename VectorType::value_type>> a_constraints(a_minlevel,
                                                                                  a_maxlevel);

    MGLevelObject<std::shared_ptr<SmootherPreconditionerType>> a_smoother_preconditioners(a_minlevel,
                                                                                        a_maxlevel);

    //
    // TODO: Generalise, maybe for operator and blockoperatorbase?
    //       Pass this part as lambda function?
    //       Or just pass vector target?
    //
    for (unsigned int level = a_minlevel; level <= a_maxlevel; level++)
      {
        const auto &dof_handler = a_dof_handlers[level];
        auto       &constraint  = a_constraints[level];

        Partitioning partitioning;
        partitioning.reinit(dof_handler);

        // ... constraints (with homogenous Dirichlet BC)
        constraint.reinit(partitioning.get_owned_dofs(), partitioning.get_relevant_dofs());

        DoFTools::make_hanging_node_constraints(dof_handler, constraint);
        // TODO: externalize this
        const Functions::ZeroFunction<dim> zero(dim);
        VectorTools::interpolate_boundary_values(mapping_collection,
                                                 dof_handler,
                                                 {{0, &zero}, {3, &zero}},
                                                 constraint);

        constraint.make_consistent_in_parallel(partitioning.get_owned_dofs(),
                                               partitioning.get_active_dofs(),
                                               partitioning.get_communicator());
        constraint.close();
        partitioning.get_relevant_dofs() = constraint.get_local_lines();

        // ... operator (just like on the finest level)
        a_operators[level] = a_block_operator.replicate();
        a_operators[level]->reinit(partitioning, dof_handler, constraint);


        // WIP: build smoother preconditioners here
        // necessary on all levels or just minlevel+1 to maxlevel?

        if constexpr (std::is_same_v<SmootherPreconditionerType, DiagonalMatrixTimer<VectorType>>)
          {
            a_smoother_preconditioners[level] =
              std::make_shared<SmootherPreconditionerType>("vmult_diagonal_ABlock");
            a_operators[level]->compute_inverse_diagonal(
              a_smoother_preconditioners[level]->get_vector());
          }
        else if constexpr (std::is_same_v<SmootherPreconditionerType, PreconditionASM<VectorType>>)
          {
            const auto patch_indices = prepare_patch_indices(dof_handler, constraint);

            if (level == a_maxlevel)
              Log::log_patch_dofs(patch_indices, dof_handler);

            // full matrix
            // TODO: this is a nasty way to get the sparsity pattern
            // so far I only created temporary sparsity patterns in the LinearAlgebra namespace,
            // but they are no longer available here
            // so for the sake of trying ASM out, I'll just create another one here
            const unsigned int myid =
              dealii::Utilities::MPI::this_mpi_process(partitioning.get_communicator());
            typename LinearAlgebra::SparsityPattern sparsity_pattern;
            sparsity_pattern.reinit(partitioning.get_owned_dofs(),
                                    partitioning.get_owned_dofs(),
                                    partitioning.get_relevant_dofs(),
                                    partitioning.get_communicator());
            DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, constraint, false, myid);
            sparsity_pattern.compress();

            a_smoother_preconditioners[level] =
              std::make_shared<SmootherPreconditionerType>(std::move(patch_indices));
            a_smoother_preconditioners[level]->initialize(a_operators[level]->get_system_matrix(),
                                                        sparsity_pattern,
                                                        partitioning);
          }
        else if constexpr (std::is_same_v<SmootherPreconditionerType,
                                          PreconditionExtendedDiagonal<VectorType>>)
          {
            const auto patch_indices = prepare_patch_indices(dof_handler, constraint);

            if (level == a_maxlevel)
              Log::log_patch_dofs(patch_indices, dof_handler);

            // full matrix
            // const unsigned int myid = dealii::Utilities::MPI::this_mpi_process(communicator);
            // DynamicSparsityPattern dsp(relevant_dofs);
            // DoFTools::make_sparsity_pattern(a_dof_handler, dsp, constraints, false, myid);
            // SparsityTools::distribute_sparsity_pattern(dsp, owned_dofs, communicator,
            // relevant_dofs);

            // reduced matrix
            AffineConstraints<double> constraints_reduced;
            constraints_reduced.reinit(partitioning.get_owned_dofs(),
                                       partitioning.get_relevant_dofs());

            const auto all_indices_relevant = extract_relevant(patch_indices, partitioning);

            std::set<types::global_dof_index> all_indices_assemble;
            reduce_constraints(constraint,
                               partitioning.get_active_dofs(),
                               all_indices_relevant,
                               constraints_reduced,
                               all_indices_assemble);

            // TODO: only works for Trilinos so far
            typename LinearAlgebra::SparsityPattern reduced_sparsity_pattern;
            reduced_sparsity_pattern.reinit(partitioning.get_owned_dofs(),
                                            partitioning.get_owned_dofs(),
                                            partitioning.get_relevant_dofs(),
                                            partitioning.get_communicator());
            make_sparsity_pattern(dof_handler,
                                  all_indices_assemble,
                                  reduced_sparsity_pattern,
                                  constraints_reduced);
            reduced_sparsity_pattern.compress();

            typename LinearAlgebra::SparseMatrix reduced_sparse_matrix;
            reduced_sparse_matrix.reinit(reduced_sparsity_pattern);
            partially_assemble_ablock(dof_handler,
                                      constraints_reduced,
                                      q_collection_v,
                                      all_indices_assemble,
                                      reduced_sparse_matrix);

            VectorType inverse_diagonal;
            a_operators[level]->compute_inverse_diagonal(inverse_diagonal);

            a_smoother_preconditioners[level] =
              std::make_shared<SmootherPreconditionerType>(std::move(patch_indices));
            // a_smoother_preconditioners[level]->initialize(mg_matrices[level]->get_system_matrix(),
            //                                             dsp,
            //                                             inverse_diagonal,
            //                                             all_indices_relevant);
            a_smoother_preconditioners[level]->initialize(reduced_sparse_matrix,
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

    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> a_transfers;
    a_transfers.resize(a_minlevel, a_maxlevel);

    // Set up intergrid operators.
    for (unsigned int level = a_minlevel; level < a_minlevel_p; ++level)
      a_transfers[level + 1].reinit_geometric_transfer(a_dof_handlers[level + 1],
                                                     a_dof_handlers[level],
                                                     a_constraints[level + 1],
                                                     a_constraints[level]);

    for (unsigned int level = a_minlevel_p; level < a_maxlevel; ++level)
      a_transfers[level + 1].reinit_polynomial_transfer(a_dof_handlers[level + 1],
                                                      a_dof_handlers[level],
                                                      a_constraints[level + 1],
                                                      a_constraints[level]);

    // Collect transfer operators within a single operator as needed by
    // the Multigrid solver class.
    MGTransferGlobalCoarsening<dim, VectorType> a_transfer(a_transfers, [&](const auto l, auto &vec) {
      a_operators[l]->initialize_dof_vector(vec);
    });

    t_mg_reinit_transfer.stop();



    TimerOutput::Scope t_mg_solve(getTimer(), "mg_solve");

    //
    // setup coarse solver
    //

    // using LevelMatrixType = StokesMatrixFree::ABlockOperator<dim, LinearAlgebra, spacedim>;
    using LevelMatrixType = OperatorType<dim, LinearAlgebra, spacedim>;
    using MGTransferType  = MGTransferGlobalCoarsening<dim, VectorType>;

    using SmootherType =
      PreconditionChebyshev<LevelMatrixType, VectorType, SmootherPreconditionerType>;
    using PreconditionerType = PreconditionMG<dim, VectorType, MGTransferType>;

    const unsigned int a_min_level = a_operators.min_level();
    const unsigned int a_max_level = a_operators.max_level();

    // Initialize level operators.
    mg::Matrix<VectorType> a_mg_matrix(a_operators); // operators = mg_matrices in mg_solver.h

    // Initialize smoothers.
    MGLevelObject<typename SmootherType::AdditionalData> a_smoother_data(a_min_level, a_max_level);

    for (unsigned int level = a_min_level; level <= a_max_level; level++)
      {
        a_smoother_data[level].preconditioner      = a_smoother_preconditioners[level];
        a_smoother_data[level].smoothing_range     = mg_data.smoother.smoothing_range;
        a_smoother_data[level].degree              = mg_data.smoother.degree;
        a_smoother_data[level].eig_cg_n_iterations = mg_data.smoother.eig_cg_n_iterations;
      }

    // ----------
    // Estimate eigenvalues on all levels, i.e., all operators
    // TODO: based on peter's code
    // https://github.com/peterrum/dealii-asm/blob/d998b9b344a19c9d2890e087f953c2f93e6546ae/include/precondition.templates.h#L292-L316
    std::vector<double> a_min_eigenvalues(a_max_level + 1, numbers::signaling_nan<double>());
    std::vector<double> a_max_eigenvalues(a_max_level + 1, numbers::signaling_nan<double>());
    if (mg_data.estimate_eigenvalues == true)
      {
        for (unsigned int level = a_min_level + 1; level <= a_max_level; level++)
          {
            SmootherType chebyshev;
            chebyshev.initialize(*a_operators[level], a_smoother_data[level]);

            VectorType vec;
            a_operators[level]->initialize_dof_vector(vec);
            const auto evs = chebyshev.estimate_eigenvalues(vec);

            a_min_eigenvalues[level] = evs.min_eigenvalue_estimate;
            a_max_eigenvalues[level] = evs.max_eigenvalue_estimate;

            // We already computed eigenvalues, reset the one in the actual smoother
            a_smoother_data[level].eig_cg_n_iterations = 0;
            a_smoother_data[level].max_eigenvalue      = evs.max_eigenvalue_estimate * 1.1;
          }

        // log maximum over all levels
        const double max = *std::max_element(++(a_max_eigenvalues.begin()), a_max_eigenvalues.end());
        getPCOut() << "   Max EV on all A MG levels:    " << max << std::endl;
        getTable().add_value("a_max_ev", max);
      }
    // ----------

    MGSmootherRelaxation<LevelMatrixType, SmootherType, VectorType> a_mg_smoother;
    a_mg_smoother.initialize(a_operators, a_smoother_data);

    // Initialize coarse-grid solver.
    ReductionControl     a_coarse_grid_solver_control(mg_data.coarse_solver.maxiter,
                                                mg_data.coarse_solver.abstol,
                                                mg_data.coarse_solver.reltol,
                                                /*log_history=*/true,
                                                /*log_result=*/true);
    SolverCG<VectorType> a_coarse_grid_solver(a_coarse_grid_solver_control);

    std::unique_ptr<MGCoarseGridBase<VectorType>> a_mg_coarse;
#ifdef DEAL_II_WITH_TRILINOS
    TrilinosWrappers::PreconditionAMG                 a_precondition_amg;
    TrilinosWrappers::PreconditionAMG::AdditionalData a_amg_data;
    a_amg_data.smoother_sweeps = mg_data.coarse_solver.smoother_sweeps;
    a_amg_data.n_cycles        = mg_data.coarse_solver.n_cycles;
    a_amg_data.smoother_type   = mg_data.coarse_solver.smoother_type.c_str();

    // CG with AMG as preconditioner
    a_precondition_amg.initialize(a_operators[a_min_level]->get_system_matrix(), a_amg_data);

    a_mg_coarse = std::make_unique<MGCoarseGridIterativeSolver<VectorType,
                                                             SolverCG<VectorType>,
                                                             LevelMatrixType,
                                                             decltype(a_precondition_amg)>>(
      a_coarse_grid_solver, *a_operators[a_min_level], a_precondition_amg);
#endif

    // Create multigrid object.
    Multigrid<VectorType> a_mg(a_mg_matrix, *a_mg_coarse, a_transfer, a_mg_smoother, a_mg_smoother);

    // ----------
    // TODO: timing based on peters dealii-multigrid
    // https://github.com/peterrum/dealii-multigrid/blob/c50581883c0dbe35c83132699e6de40da9b1b255/multigrid_throughput.cc#L1183-L1192
    std::vector<std::vector<std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>>
      all_mg_timers(a_max_level - a_min_level + 1);

    for (unsigned int i = 0; i < all_mg_timers.size(); ++i)
      all_mg_timers[i].resize(7);

    if (mg_data.log_levels == true)
      {
        const auto create_mg_timer_function = [&](const unsigned int i, const std::string &label) {
          return [i, label, &all_mg_timers](const bool flag, const unsigned int level) {
            // if (false && flag)
            //   std::cout << label << " " << level << std::endl;
            if (flag)
              all_mg_timers[level][i].second = std::chrono::system_clock::now();
            else
              all_mg_timers[level][i].first +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::system_clock::now() - all_mg_timers[level][i].second)
                  .count() /
                1e9;
          };
        };

        a_mg.connect_pre_smoother_step(create_mg_timer_function(0, "pre_smoother_step"));
        a_mg.connect_residual_step(create_mg_timer_function(1, "residual_step"));
        a_mg.connect_restriction(create_mg_timer_function(2, "restriction"));
        a_mg.connect_coarse_solve(create_mg_timer_function(3, "coarse_solve"));
        a_mg.connect_prolongation(create_mg_timer_function(4, "prolongation"));
        a_mg.connect_edge_prolongation(create_mg_timer_function(5, "edge_prolongation"));
        a_mg.connect_post_smoother_step(create_mg_timer_function(6, "post_smoother_step"));
      }
    // ----------

    // Convert it to a preconditioner.
    PreconditionerType a_block_preconditioner(a_dof_handler, a_mg, a_transfer);










    // build hp-MG for Schur block
    const DoFHandler<dim, spacedim> &schur_dof_handler = *(stokes_dof_handlers[1]);

    MGLevelObject<DoFHandler<dim, spacedim>>                                   schur_dof_handlers;
    MGLevelObject<std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>> schur_operators;

    const unsigned int schur_n_p_levels =
      MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
        get_max_active_fe_degree(schur_dof_handler), mg_data.transfer.p_sequence)
        .size();

    std::map<unsigned int, unsigned int> schur_fe_index_for_degree;
    // Skip the first "dummy element"
    for (unsigned int i = 1; i < schur_dof_handler.get_fe_collection().size(); ++i)
      {
        const unsigned int degree = schur_dof_handler.get_fe(i).degree;
        Assert(schur_fe_index_for_degree.find(degree) == schur_fe_index_for_degree.end(),
               ExcMessage("FECollection does not contain unique degrees."));
        schur_fe_index_for_degree[degree] = i;
      }
    
    unsigned int schur_minlevel   = 0;
    unsigned int schur_minlevel_p = n_h_levels;
    unsigned int schur_maxlevel   = n_h_levels + schur_n_p_levels - 1;

    schur_dof_handlers.resize(schur_minlevel, schur_maxlevel);
    schur_operators.resize(schur_minlevel, schur_maxlevel);

    for (unsigned int l = 0; l < n_h_levels; ++l)
      {
        schur_dof_handlers[l].reinit(*coarse_grid_triangulations[l]);
        schur_dof_handlers[l].distribute_dofs(schur_dof_handler.get_fe_collection());
      }
    
    for (unsigned int i = 0, l = schur_maxlevel; i < schur_n_p_levels; ++i, --l)
      {
        schur_dof_handlers[l].reinit(schur_dof_handler.get_triangulation());

        if (l == schur_maxlevel) // finest level
          {
            auto &dof_handler_mg = schur_dof_handlers[l];

            auto cell_other = schur_dof_handler.begin_active();
            for (auto &cell : dof_handler_mg.active_cell_iterators())
              {
                if (cell->is_locally_owned())
                  cell->set_active_fe_index(cell_other->active_fe_index());
                cell_other++;
              }
          }
        else // coarse level
          {
            auto &dof_handler_fine   = schur_dof_handlers[l + 1];
            auto &dof_handler_coarse = schur_dof_handlers[l + 0];

            auto cell_other = dof_handler_fine.begin_active();
            for (auto &cell : dof_handler_coarse.active_cell_iterators())
              {
                if (cell->is_locally_owned())
                  {
                    const unsigned int next_degree =
                      MGTransferGlobalCoarseningTools::create_next_polynomial_coarsening_degree(
                        cell_other->get_fe().degree, mg_data.transfer.p_sequence);
                    Assert(schur_fe_index_for_degree.find(next_degree) != schur_fe_index_for_degree.end(),
                           ExcMessage("Next polynomial degree in sequence "
                                      "does not exist in FECollection."));

                    cell->set_active_fe_index(schur_fe_index_for_degree[next_degree]);
                  }
                cell_other++;
              }
          }

        schur_dof_handlers[l].distribute_dofs(schur_dof_handler.get_fe_collection());
      }

    MGLevelObject<AffineConstraints<typename VectorType::value_type>> schur_constraints(schur_minlevel,
                                                                                  schur_maxlevel);

    MGLevelObject<std::shared_ptr<SmootherPreconditionerType>> schur_smoother_preconditioners(schur_minlevel,
                                                                                        schur_maxlevel);

    for (unsigned int level = schur_minlevel; level <= schur_maxlevel; level++)
      {
        const auto &dof_handler = schur_dof_handlers[level];
        auto       &constraint  = schur_constraints[level];

        Partitioning partitioning;
        partitioning.reinit(dof_handler);

        // ... constraints (with homogenous Dirichlet BC)
        constraint.reinit(partitioning.get_owned_dofs(), partitioning.get_relevant_dofs());

        DoFTools::make_hanging_node_constraints(dof_handler, constraint);
        // no boundary conditions on pressure/Schur

        constraint.make_consistent_in_parallel(partitioning.get_owned_dofs(),
                                               partitioning.get_active_dofs(),
                                               partitioning.get_communicator());
        constraint.close();
        partitioning.get_relevant_dofs() = constraint.get_local_lines();

        // ... operator (just like on the finest level)
        schur_operators[level] = schur_block_operator.replicate();
        schur_operators[level]->reinit(partitioning, dof_handler, constraint);

        // smoother preconditioners
        if constexpr (std::is_same_v<SmootherPreconditionerType, DiagonalMatrixTimer<VectorType>>)
          {
            schur_smoother_preconditioners[level] =
              std::make_shared<SmootherPreconditionerType>("vmult_diagonal_SchurBlock");
            schur_operators[level]->compute_inverse_diagonal(
              schur_smoother_preconditioners[level]->get_vector());
          }
        else if constexpr (std::is_same_v<SmootherPreconditionerType, PreconditionASM<VectorType>>)
          {
            const auto patch_indices = prepare_patch_indices(dof_handler, constraint);

            // TODO: don't log schur for now
            // if (level == schur_maxlevel)
            //   Log::log_patch_dofs(patch_indices, dof_handler);

            // full matrix
            // TODO: this is a nasty way to get the sparsity pattern
            // so far I only created temporary sparsity patterns in the LinearAlgebra namespace,
            // but they are no longer available here
            // so for the sake of trying ASM out, I'll just create another one here
            const unsigned int myid =
              dealii::Utilities::MPI::this_mpi_process(partitioning.get_communicator());
            typename LinearAlgebra::SparsityPattern sparsity_pattern;
            sparsity_pattern.reinit(partitioning.get_owned_dofs(),
                                    partitioning.get_owned_dofs(),
                                    partitioning.get_relevant_dofs(),
                                    partitioning.get_communicator());
            DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, constraint, false, myid);
            sparsity_pattern.compress();

            schur_smoother_preconditioners[level] =
              std::make_shared<SmootherPreconditionerType>(std::move(patch_indices));
            schur_smoother_preconditioners[level]->initialize(schur_operators[level]->get_system_matrix(),
                                                        sparsity_pattern,
                                                        partitioning);
          }
        else if constexpr (std::is_same_v<SmootherPreconditionerType,
                                          PreconditionExtendedDiagonal<VectorType>>)
        {
          const auto patch_indices = prepare_patch_indices(dof_handler, constraint);

          // TODO: don't log schur for now
          // if (level == schur_maxlevel)
          //   Log::log_patch_dofs(patch_indices, dof_handler);

          // reduced matrix
          AffineConstraints<double> constraints_reduced;
          constraints_reduced.reinit(partitioning.get_owned_dofs(),
                                      partitioning.get_relevant_dofs());

          const auto all_indices_relevant = extract_relevant(patch_indices, partitioning);

          std::set<types::global_dof_index> all_indices_assemble;
          reduce_constraints(constraint,
                              partitioning.get_active_dofs(),
                              all_indices_relevant,
                              constraints_reduced,
                              all_indices_assemble);

          // TODO: only works for Trilinos so far
          typename LinearAlgebra::SparsityPattern reduced_sparsity_pattern;
          reduced_sparsity_pattern.reinit(partitioning.get_owned_dofs(),
                                          partitioning.get_owned_dofs(),
                                          partitioning.get_relevant_dofs(),
                                          partitioning.get_communicator());
          make_sparsity_pattern(dof_handler,
                                all_indices_assemble,
                                reduced_sparsity_pattern,
                                constraints_reduced);
          reduced_sparsity_pattern.compress();

          typename LinearAlgebra::SparseMatrix reduced_sparse_matrix;
          reduced_sparse_matrix.reinit(reduced_sparsity_pattern);
          partially_assemble_schurblock(dof_handler,
                                    constraints_reduced,
                                    q_collection_v, // TODO: no need
                                    all_indices_assemble,
                                    reduced_sparse_matrix);

          VectorType inverse_diagonal;
          schur_operators[level]->compute_inverse_diagonal(inverse_diagonal);

          // TODO: change to template 
          schur_smoother_preconditioners[level] =
            std::make_shared<SmootherPreconditionerType>(std::move(patch_indices));
          schur_smoother_preconditioners[level]->initialize(reduced_sparse_matrix,
                                                      reduced_sparsity_pattern,
                                                      inverse_diagonal,
                                                      all_indices_relevant);
        }
      }
    
    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> schur_transfers;
    schur_transfers.resize(schur_minlevel, schur_maxlevel);

    // Set up intergrid operators.
    for (unsigned int level = schur_minlevel; level < schur_minlevel_p; ++level)
      schur_transfers[level + 1].reinit_geometric_transfer(schur_dof_handlers[level + 1],
                                                     schur_dof_handlers[level],
                                                     schur_constraints[level + 1],
                                                     schur_constraints[level]);

    for (unsigned int level = schur_minlevel_p; level < schur_maxlevel; ++level)
      schur_transfers[level + 1].reinit_polynomial_transfer(schur_dof_handlers[level + 1],
                                                      schur_dof_handlers[level],
                                                      schur_constraints[level + 1],
                                                      schur_constraints[level]);

    MGTransferGlobalCoarsening<dim, VectorType> schur_transfer(schur_transfers, [&](const auto l, auto &vec) {
      schur_operators[l]->initialize_dof_vector(vec);
    });

    // setup coarse solver
    const unsigned int schur_min_level = schur_operators.min_level();
    const unsigned int schur_max_level = schur_operators.max_level();

    mg::Matrix<VectorType> schur_mg_matrix(schur_operators);

    MGLevelObject<typename SmootherType::AdditionalData> schur_smoother_data(schur_min_level, schur_max_level);

    for (unsigned int level = schur_min_level; level <= schur_max_level; level++)
      {
        schur_smoother_data[level].preconditioner      = schur_smoother_preconditioners[level];
        schur_smoother_data[level].smoothing_range     = mg_data.smoother.smoothing_range;
        schur_smoother_data[level].degree              = mg_data.smoother.degree;
        schur_smoother_data[level].eig_cg_n_iterations = mg_data.smoother.eig_cg_n_iterations;
      }
    
    std::vector<double> schur_min_eigenvalues(schur_max_level + 1, numbers::signaling_nan<double>());
    std::vector<double> schur_max_eigenvalues(schur_max_level + 1, numbers::signaling_nan<double>());
    if (mg_data.estimate_eigenvalues == true)
      {
        for (unsigned int level = schur_min_level + 1; level <= schur_max_level; level++)
          {
            SmootherType chebyshev;
            chebyshev.initialize(*schur_operators[level], schur_smoother_data[level]);

            VectorType vec;
            schur_operators[level]->initialize_dof_vector(vec);
            const auto evs = chebyshev.estimate_eigenvalues(vec);

            schur_min_eigenvalues[level] = evs.min_eigenvalue_estimate;
            schur_max_eigenvalues[level] = evs.max_eigenvalue_estimate;

            schur_smoother_data[level].eig_cg_n_iterations = 0;
            schur_smoother_data[level].max_eigenvalue      = evs.max_eigenvalue_estimate * 1.1;
          }

        const double max = *std::max_element(++(schur_max_eigenvalues.begin()), schur_max_eigenvalues.end());
        getPCOut() << "   Max EV on all Schur MG levels:    " << max << std::endl;
        getTable().add_value("schur_max_ev", max);
      }
    
    MGSmootherRelaxation<LevelMatrixType, SmootherType, VectorType> schur_mg_smoother;
    schur_mg_smoother.initialize(schur_operators, schur_smoother_data);

    ReductionControl     schur_coarse_grid_solver_control(mg_data.coarse_solver.maxiter,
                                                mg_data.coarse_solver.abstol,
                                                mg_data.coarse_solver.reltol,
                                                /*log_history=*/true,
                                                /*log_result=*/true);
    SolverCG<VectorType> schur_coarse_grid_solver(schur_coarse_grid_solver_control);

    std::unique_ptr<MGCoarseGridBase<VectorType>> schur_mg_coarse;
#ifdef DEAL_II_WITH_TRILINOS
    TrilinosWrappers::PreconditionAMG                 schur_precondition_amg;
    TrilinosWrappers::PreconditionAMG::AdditionalData schur_amg_data;
    schur_amg_data.smoother_sweeps = mg_data.coarse_solver.smoother_sweeps;
    schur_amg_data.n_cycles        = mg_data.coarse_solver.n_cycles;
    schur_amg_data.smoother_type   = mg_data.coarse_solver.smoother_type.c_str();

    schur_precondition_amg.initialize(schur_operators[schur_min_level]->get_system_matrix(), schur_amg_data);

    schur_mg_coarse = std::make_unique<MGCoarseGridIterativeSolver<VectorType,
                                                             SolverCG<VectorType>,
                                                             LevelMatrixType,
                                                             decltype(schur_precondition_amg)>>(
      schur_coarse_grid_solver, *schur_operators[schur_min_level], schur_precondition_amg);
#endif

    Multigrid<VectorType> schur_mg(schur_mg_matrix, *schur_mg_coarse, schur_transfer, schur_mg_smoother, schur_mg_smoother);

    PreconditionerType schur_block_preconditioner(schur_dof_handler, schur_mg, schur_transfer);
    









    const BlockSchurPreconditioner<LinearAlgebra,
                                   StokesMatrixFree::StokesOperator<dim, LinearAlgebra, spacedim>,
                                   OperatorType<dim, LinearAlgebra, spacedim>,
                                   OperatorType<dim, LinearAlgebra, spacedim>,
                                   PreconditionerType,
                                   PreconditionerType>
      preconditioner(stokes_operator,
                     a_block_operator,
                     schur_block_operator,
                     a_block_preconditioner,
                     schur_block_preconditioner,
                     /*do_solve_A=*/false,
                     /*do_solve_Schur_complement=*/false);

    // set up solver
    dealii::PrimitiveVectorMemory<typename LinearAlgebra::BlockVector> mem;

    typename dealii::SolverGMRES<typename LinearAlgebra::BlockVector>::AdditionalData gmres_data(
      50);
    dealii::SolverGMRES<typename LinearAlgebra::BlockVector> solver(solver_control_refined,
                                                                     mem,
                                                                     gmres_data);

    solver.solve(stokes_operator, dst, src, preconditioner);

    // ----------
    // dump to Table and then file system
    if (mg_data.log_levels == true)
      {
        std::vector<std::vector<Utilities::MPI::MinMaxAvg>> min_max_avg(all_mg_timers.size());
        for (unsigned int level = 0; level < all_mg_timers.size(); ++level)
          {
            min_max_avg[level].resize(7);
            for (unsigned int i = 0; i < 7; ++i)
              min_max_avg[level][i] = Utilities::MPI::min_max_avg(all_mg_timers[level][i].first,
                                                                  a_dof_handler.get_communicator());
          }

        if (Utilities::MPI::this_mpi_process(a_dof_handler.get_communicator()) == 0)
          {
            dealii::ConvergenceTable table;
            for (unsigned int level = 0; level < all_mg_timers.size(); ++level)
              {
                table.add_value("level", level);
                table.add_value("active_cells",
                                a_dof_handlers[level].get_triangulation().n_global_active_cells());
                table.add_value("dofs", a_dof_handlers[level].n_dofs());
                table.add_value("pre_smoother_step_min", min_max_avg[level][0].min);
                table.add_value("pre_smoother_step_max", min_max_avg[level][0].max);
                table.add_value("pre_smoother_step_avg", min_max_avg[level][0].avg);
                table.add_value("residual_step_min", min_max_avg[level][1].min);
                table.add_value("residual_step_max", min_max_avg[level][1].max);
                table.add_value("residual_step_avg", min_max_avg[level][1].avg);
                table.add_value("restriction_min", min_max_avg[level][2].min);
                table.add_value("restriction_max", min_max_avg[level][2].max);
                table.add_value("restriction_avg", min_max_avg[level][2].avg);
                table.add_value("coarse_solve_min", min_max_avg[level][3].min);
                table.add_value("coarse_solve_max", min_max_avg[level][3].max);
                table.add_value("coarse_solve_avg", min_max_avg[level][3].avg);
                table.add_value("prolongation_min", min_max_avg[level][4].min);
                table.add_value("prolongation_max", min_max_avg[level][4].max);
                table.add_value("prolongation_avg", min_max_avg[level][4].avg);
                table.add_value("edge_prolongation_min", min_max_avg[level][5].min);
                table.add_value("edge_prolongation_max", min_max_avg[level][5].max);
                table.add_value("edge_prolongation_avg", min_max_avg[level][5].avg);
                table.add_value("post_smoother_step_min", min_max_avg[level][6].min);
                table.add_value("post_smoother_step_max", min_max_avg[level][6].max);
                table.add_value("post_smoother_step_avg", min_max_avg[level][6].avg);
                if (mg_data.estimate_eigenvalues == true)
                  {
                    table.add_value("a_min_eigenvalue", a_min_eigenvalues[level]);
                    table.add_value("a_max_eigenvalue", a_max_eigenvalues[level]);
                  }
              }

            std::ofstream mg_level_stream(filename_mg_level);
            table.write_text(mg_level_stream);
          }
      }
    // ----------

    t_mg_solve.stop();
  }
} // namespace StokesMatrixFree


#endif
