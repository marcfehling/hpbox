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


/*
 * NOTE:
 * This file is copied from WIP PR #11699 which determines the interface
 * of the MGSolverOperatorBase class.
 */



#ifndef dealii_mg_solver_h
#define dealii_mg_solver_h


#include <deal.II/base/config.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/signaling_nan.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix_tools.h>

#ifdef DEAL_II_WITH_TRILINOS
#  include <deal.II/lac/trilinos_precondition.h>
// #  include <deal.II/lac/trilinos_sparse_matrix.h>
#endif

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <linear_algebra.h>

#include <vector>

DEAL_II_NAMESPACE_OPEN



/**
 * Adopted from
 * https://github.com/peterrum/dealii-asm/blob/d998b9b344a19c9d2890e087f953c2f93e6546ae/include/preconditioners.h#L145.
 */
template <typename Number, int dim, int spacedim>
class PreconditionASM
{
private:
  enum class WeightingType
  {
    none,
    left,
    right,
    symm
  };

public:
  PreconditionASM(const DoFHandler<dim, spacedim> &dof_handler)
    : dof_handler(dof_handler)
    , weighting_type(WeightingType::symm)
  {}

  template <typename GlobalSparseMatrixType, typename GlobalSparsityPattern>
  void
  initialize(const GlobalSparseMatrixType &global_sparse_matrix,
             const GlobalSparsityPattern  &global_sparsity_pattern)
  {
    SparseMatrixTools::restrict_to_cells(global_sparse_matrix,
                                         global_sparsity_pattern,
                                         dof_handler,
                                         blocks);

    for (auto &block : blocks)
      if (block.m() > 0 && block.n() > 0)
        block.gauss_jordan();
  }

  template <typename VectorType>
  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    dst = 0.0;
    src.update_ghost_values();

    Vector<double> vector_src, vector_dst, vector_weights;

    VectorType weights;

    if (weighting_type != WeightingType::none)
      {
        weights.reinit(src);

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned() == false)
              continue;

            const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
            vector_weights.reinit(dofs_per_cell);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              vector_weights[i] = 1.0;

            cell->distribute_local_to_global(vector_weights, weights);
          }

        weights.compress(VectorOperation::add);
        for (auto &i : weights)
          i = (weighting_type == WeightingType::symm) ? std::sqrt(1.0 / i) :
                                                        (1.0 / i);
        weights.update_ghost_values();
      }

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();

        vector_src.reinit(dofs_per_cell);
        vector_dst.reinit(dofs_per_cell);
        if (weighting_type != WeightingType::none)
          vector_weights.reinit(dofs_per_cell);

        cell->get_dof_values(src, vector_src);
        if (weighting_type != WeightingType::none)
          cell->get_dof_values(weights, vector_weights);

        if (weighting_type == WeightingType::symm ||
            weighting_type == WeightingType::right)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            vector_src[i] *= vector_weights[i];

        blocks[cell->active_cell_index()].vmult(vector_dst, vector_src);

        if (weighting_type == WeightingType::symm ||
            weighting_type == WeightingType::left)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            vector_dst[i] *= vector_weights[i];

        cell->distribute_local_to_global(vector_dst, dst);
      }

    src.zero_out_ghost_values();
    dst.compress(VectorOperation::add);
  }

private:
  const DoFHandler<dim, spacedim> &dof_handler;
  std::vector<FullMatrix<Number>>  blocks;

  const WeightingType weighting_type;
};



// struct MGSolverParameters
//{
//  struct CoarseSolverParameters
//  {
//    std::string  type            = "cg_with_amg";
//    unsigned int maxiter         = 10000;
//    double       abstol          = 1e-20;
//    double       reltol          = 1e-4;
//    unsigned int smoother_sweeps = 1;
//    unsigned int n_cycles        = 1;
//    std::string  smoother_type   = "ILU";
//  };

//  struct SmootherParameters
//  {
//    std::string  type                = "chebyshev";
//    double       smoothing_range     = 20;
//    unsigned int degree              = 5;
//    unsigned int eig_cg_n_iterations = 20;
//  };

//  SmootherParameters     smoother;
//  CoarseSolverParameters coarse_solver;
//};
struct MGSolverParameters
{
  struct
  {
    std::string  type            = "cg_with_amg";
    unsigned int maxiter         = 10000;
    double       abstol          = 1e-20;
    double       reltol          = 1e-4;
    unsigned int smoother_sweeps = 1;
    unsigned int n_cycles        = 1;
    std::string  smoother_type   = "ILU";
  } coarse_solver;

  struct
  {
    std::string  type                = "chebyshev";
    double       smoothing_range     = 20;
    unsigned int degree              = 5;
    unsigned int eig_cg_n_iterations = 20;
  } smoother;

  struct
  {
    MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType p_sequence =
      MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::decrease_by_one;
    bool perform_h_transfer = true;
  } transfer;
};



// template <int dim_, typename number>
// class MGSolverOperatorBase : public Subscriptor
//{
// public:
//  static const int dim = dim_;
//  using value_type     = number;
//  using VectorType     = LinearAlgebra::distributed::Vector<number>;
template <int dim, typename VectorType, typename MatrixType>
class MGSolverOperatorBase : public Subscriptor
{
public:
  using value_type = typename VectorType::value_type;

  // Return number of rows of the matrix. Since we are dealing with a
  // symmetrical matrix, the returned value is the same as the number of
  // columns.
  virtual types::global_dof_index
  m() const;

  // Access a particular element in the matrix. This function is neither
  // needed nor implemented, however, is required to compile the program.
  virtual value_type
  el(unsigned int, unsigned int) const;

  // Allocate memory for a distributed vector.
  virtual void
  initialize_dof_vector(VectorType &vec) const;

  // Perform an operator application on the vector @p src.
  virtual void
  vmult(VectorType &dst, const VectorType &src) const;

  // Perform the transposed operator evaluation. Since we are considering
  // symmetric matrices, this function is identical to the above function.
  virtual void
  Tvmult(VectorType &dst, const VectorType &src) const;

  // Compute the inverse of the diagonal of the vector and store it into the
  // provided vector. The inverse diagonal is used below in a Chebyshev
  // smoother.
  virtual void
  compute_inverse_diagonal(VectorType &diagonal) const;

  // Return the actual system matrix, which can be used in any matrix-based
  // solvers (like AMG).
  virtual const MatrixType &
  get_system_matrix() const;

private:
  const MatrixType dummy_sparse_matrix;
};



// template <int dim_, typename number>
// types::global_dof_index
// MGSolverOperatorBase<dim_, number>::m() const
template <int dim, typename VectorType, typename MatrixType>
types::global_dof_index
MGSolverOperatorBase<dim, VectorType, MatrixType>::m() const
{
  Assert(false, ExcNotImplemented());
  return 0;
}



// template <int dim_, typename number>
// number
// MGSolverOperatorBase<dim_, number>::el(unsigned int, unsigned int) const
template <int dim, typename VectorType, typename MatrixType>
typename MGSolverOperatorBase<dim, VectorType, MatrixType>::value_type
MGSolverOperatorBase<dim, VectorType, MatrixType>::el(unsigned int, unsigned int) const
{
  Assert(false, ExcNotImplemented());
  return 0;
}



// template <int dim_, typename number>
// void
// MGSolverOperatorBase<dim_, number>::initialize_dof_vector(VectorType &vec)
// const
template <int dim, typename VectorType, typename MatrixType>
void
MGSolverOperatorBase<dim, VectorType, MatrixType>::initialize_dof_vector(VectorType &vec) const
{
  Assert(false, ExcNotImplemented());
  (void)vec;
}



// template <int dim_, typename number>
// void
// MGSolverOperatorBase<dim_, number>::vmult(VectorType &      dst,
//                                          const VectorType &src) const
template <int dim, typename VectorType, typename MatrixType>
void
MGSolverOperatorBase<dim, VectorType, MatrixType>::vmult(VectorType       &dst,
                                                         const VectorType &src) const
{
  Assert(false, ExcNotImplemented());
  (void)dst;
  (void)src;
}



// template <int dim_, typename number>
// void
// MGSolverOperatorBase<dim_, number>::Tvmult(VectorType &      dst,
//                                           const VectorType &src) const
template <int dim, typename VectorType, typename MatrixType>
void
MGSolverOperatorBase<dim, VectorType, MatrixType>::Tvmult(VectorType       &dst,
                                                          const VectorType &src) const
{
  Assert(false, ExcNotImplemented());
  (void)dst;
  (void)src;
}



// template <int dim_, typename number>
// void
// MGSolverOperatorBase<dim_, number>::compute_inverse_diagonal(
//  VectorType &diagonal) const
template <int dim, typename VectorType, typename MatrixType>
void
MGSolverOperatorBase<dim, VectorType, MatrixType>::compute_inverse_diagonal(
  VectorType &diagonal) const
{
  Assert(false, ExcNotImplemented());
  (void)diagonal;
}



// template <int dim_, typename number>
// const TrilinosWrappers::SparseMatrix &
// MGSolverOperatorBase<dim_, number>::get_system_matrix() const
template <int dim, typename VectorType, typename MatrixType>
const MatrixType &
MGSolverOperatorBase<dim, VectorType, MatrixType>::get_system_matrix() const
{
  Assert(false, ExcNotImplemented());
  return dummy_sparse_matrix;
}



// ----- mg_solve -----

template <typename VectorType,
          int dim,
          typename SystemMatrixType,
          typename LevelMatrixType,
          typename MGTransferType>
static void
mg_solve(SolverControl                                         &solver_control,
         VectorType                                            &dst,
         const VectorType                                      &src,
         const MGSolverParameters                              &mg_data,
         const DoFHandler<dim>                                 &dof,
         const SystemMatrixType                                &fine_matrix,
         const MGLevelObject<std::unique_ptr<LevelMatrixType>> &mg_matrices,
         const MGLevelObject<DoFHandler<dim>>                  &mg_dof_handlers,
         const MGLevelObject<AffineConstraints<double>>        &mg_constraints,
         const MGTransferType                                  &mg_transfer,
         const std::string                                     &filename_mg_level)
{
  AssertThrow(mg_data.smoother.type == "chebyshev", ExcNotImplemented());

  const unsigned int min_level = mg_matrices.min_level();
  const unsigned int max_level = mg_matrices.max_level();

  // using value_type                 = typename VectorType::value_type;
  // using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherPreconditionerType = PreconditionASM<double, dim, dim>;
  using SmootherType =
    PreconditionChebyshev<LevelMatrixType, VectorType, SmootherPreconditionerType>;
  using PreconditionerType = PreconditionMG<dim, VectorType, MGTransferType>;

  // Initialize level operators.
  mg::Matrix<VectorType> mg_matrix(mg_matrices);

  // Initialize smoothers.
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level, max_level);

  for (unsigned int level = min_level; level <= max_level; level++)
    {
      //smoother_data[level].preconditioner = std::make_shared<SmootherPreconditionerType>();
      //mg_matrices[level]->compute_inverse_diagonal(
      //  smoother_data[level].preconditioner->get_vector());

      // ----------
      // TODO: this is a nasty way to get the sparsity pattern
      // so far I only created temporary sparsity patterns in the LinearAlgebra namespace,
      // but they are no longer available here
      // so for the sake of trying ASM out, I'll just create another one here
      const auto &dof_handler = mg_dof_handlers[level];
      const auto &constraints = mg_constraints[level];

      auto communicator = dof_handler.get_communicator();
      const auto owned_dofs = dof_handler.locally_owned_dofs();
      const IndexSet relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
      const unsigned int myid = dealii::Utilities::MPI::this_mpi_process(communicator);

      DynamicSparsityPattern dsp(relevant_dofs);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false, myid);
      SparsityTools::distribute_sparsity_pattern(dsp, owned_dofs, communicator, relevant_dofs);

      smoother_data[level].preconditioner = std::make_shared<SmootherPreconditionerType>(mg_dof_handlers[level]);
      smoother_data[level].preconditioner->initialize(mg_matrices[level]->get_system_matrix(), dsp);
      // ----------

      smoother_data[level].smoothing_range     = mg_data.smoother.smoothing_range;
      smoother_data[level].degree              = mg_data.smoother.degree;
      smoother_data[level].eig_cg_n_iterations = mg_data.smoother.eig_cg_n_iterations;
    }

  // ----------
  // Estimate eigenvalues on all levels, i.e., all operators
  // TODO: based on peter's code
  // https://github.com/peterrum/dealii-asm/blob/d998b9b344a19c9d2890e087f953c2f93e6546ae/include/precondition.templates.h#L292-L316
  std::vector<double> min_eigenvalues(max_level + 1, numbers::signaling_nan<double>());
  std::vector<double> max_eigenvalues(max_level + 1, numbers::signaling_nan<double>());
  for (unsigned int level = min_level + 1; level <= max_level; level++)
    {
      SmootherType chebyshev;
      chebyshev.initialize(*mg_matrices[level], smoother_data[level]);

      VectorType vec;
      mg_matrices[level]->initialize_dof_vector(vec);
      const auto evs = chebyshev.estimate_eigenvalues(vec);

      min_eigenvalues[level] = evs.min_eigenvalue_estimate;
      max_eigenvalues[level] = evs.max_eigenvalue_estimate;

      // We already computed eigenvalues, reset the one in the actual smoother
      smoother_data[level].eig_cg_n_iterations = 0;
      smoother_data[level].max_eigenvalue = evs.max_eigenvalue_estimate * 1.1;
    }
  // ----------

  MGSmootherRelaxation<LevelMatrixType, SmootherType, VectorType> mg_smoother;
  mg_smoother.initialize(mg_matrices, smoother_data);

  // Initialize coarse-grid solver.
  ReductionControl     coarse_grid_solver_control(mg_data.coarse_solver.maxiter,
                                              mg_data.coarse_solver.abstol,
                                              mg_data.coarse_solver.reltol,
                                              /*log_history=*/true,
                                              /*log_result=*/true);
  SolverCG<VectorType> coarse_grid_solver(coarse_grid_solver_control);

  PreconditionIdentity precondition_identity;
  PreconditionChebyshev<LevelMatrixType, VectorType, DiagonalMatrix<VectorType>>
    precondition_chebyshev;

#ifdef DEAL_II_WITH_TRILINOS
  TrilinosWrappers::PreconditionAMG precondition_amg;
#endif

  std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  if (mg_data.coarse_solver.type == "cg")
    {
      // CG with identity matrix as preconditioner

      mg_coarse =
        std::make_unique<MGCoarseGridIterativeSolver<VectorType,
                                                     SolverCG<VectorType>,
                                                     LevelMatrixType,
                                                     PreconditionIdentity>>(coarse_grid_solver,
                                                                            *mg_matrices[min_level],
                                                                            precondition_identity);
    }
  else if (mg_data.coarse_solver.type == "cg_with_chebyshev")
    {
      // CG with Chebyshev as preconditioner

      Assert(false, ExcNotImplemented());

//      typename SmootherType::AdditionalData smoother_data;
//
//      smoother_data.preconditioner = std::make_shared<DiagonalMatrix<VectorType>>();
//      mg_matrices[min_level]->compute_inverse_diagonal(smoother_data.preconditioner->get_vector());
//      smoother_data.smoothing_range     = mg_data.smoother.smoothing_range;
//      smoother_data.degree              = mg_data.smoother.degree;
//      smoother_data.eig_cg_n_iterations = mg_data.smoother.eig_cg_n_iterations;
//
//      precondition_chebyshev.initialize(*mg_matrices[min_level], smoother_data);
//
//      mg_coarse = std::make_unique<MGCoarseGridIterativeSolver<VectorType,
//                                                               SolverCG<VectorType>,
//                                                               LevelMatrixType,
//                                                               decltype(precondition_chebyshev)>>(
//        coarse_grid_solver, *mg_matrices[min_level], precondition_chebyshev);
    }
  else if (mg_data.coarse_solver.type == "cg_with_amg")
    {
      // CG with AMG as preconditioner

#ifdef DEAL_II_WITH_TRILINOS
      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.smoother_sweeps = mg_data.coarse_solver.smoother_sweeps;
      amg_data.n_cycles        = mg_data.coarse_solver.n_cycles;
      amg_data.smoother_type   = mg_data.coarse_solver.smoother_type.c_str();

      // CG with AMG as preconditioner
      precondition_amg.initialize(mg_matrices[min_level]->get_system_matrix(), amg_data);

      mg_coarse = std::make_unique<MGCoarseGridIterativeSolver<VectorType,
                                                               SolverCG<VectorType>,
                                                               LevelMatrixType,
                                                               decltype(precondition_amg)>>(
        coarse_grid_solver, *mg_matrices[min_level], precondition_amg);
#else
      AssertThrow(false, ExcNotImplemented());
#endif
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  // Create multigrid object.
  Multigrid<VectorType> mg(mg_matrix, *mg_coarse, mg_transfer, mg_smoother, mg_smoother);

  // ----------
  // TODO: timing based on peters dealii-multigrid
  // https://github.com/peterrum/dealii-multigrid/blob/c50581883c0dbe35c83132699e6de40da9b1b255/multigrid_throughput.cc#L1183-L1192
  std::vector<std::vector<std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>>
    all_mg_timers(max_level - min_level + 1);

  for (unsigned int i = 0; i < all_mg_timers.size(); ++i)
    all_mg_timers[i].resize(7);

  const auto create_mg_timer_function = [&](const unsigned int i, const std::string &label) {
    return [i, label, &all_mg_timers](const bool flag, const unsigned int level) {
      // if (false && flag)
      //   std::cout << label << " " << level << std::endl;
      if (flag)
        all_mg_timers[level][i].second = std::chrono::system_clock::now();
      else
        all_mg_timers[level][i].first +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() -
                                                                all_mg_timers[level][i].second)
            .count() /
          1e9;
    };
  };

  mg.connect_pre_smoother_step(create_mg_timer_function(0, "pre_smoother_step"));
  mg.connect_residual_step(create_mg_timer_function(1, "residual_step"));
  mg.connect_restriction(create_mg_timer_function(2, "restriction"));
  mg.connect_coarse_solve(create_mg_timer_function(3, "coarse_solve"));
  mg.connect_prolongation(create_mg_timer_function(4, "prolongation"));
  mg.connect_edge_prolongation(create_mg_timer_function(5, "edge_prolongation"));
  mg.connect_post_smoother_step(create_mg_timer_function(6, "post_smoother_step"));
  // ----------

  // Convert it to a preconditioner.
  PreconditionerType preconditioner(dof, mg, mg_transfer);

  // Finally, solve.
  SolverCG<VectorType>(solver_control).solve(fine_matrix, dst, src, preconditioner);


  // ----------
  // dump to Table and then file system
  if (Utilities::MPI::this_mpi_process(dof.get_communicator()) == 0)
    {
      dealii::ConvergenceTable table;
      for (unsigned int level = 0; level < all_mg_timers.size(); ++level)
        {
          table.add_value("level", level);
          table.add_value("pre_smoother_step", all_mg_timers[level][0].first);
          table.add_value("residual_step", all_mg_timers[level][1].first);
          table.add_value("restriction", all_mg_timers[level][2].first);
          table.add_value("coarse_solve", all_mg_timers[level][3].first);
          table.add_value("prolongation", all_mg_timers[level][4].first);
          table.add_value("edge_prolongation", all_mg_timers[level][5].first);
          table.add_value("post_smoother_step", all_mg_timers[level][6].first);
          table.add_value("min_eigenvalue", min_eigenvalues[level]);
          table.add_value("max_eigenvalue", max_eigenvalues[level]);
        }
      std::ofstream mg_level_stream(filename_mg_level);
      table.write_text(mg_level_stream);
    }
  // ----------
}

#ifndef DOXYGEN



#endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE

#endif
