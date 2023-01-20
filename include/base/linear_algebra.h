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

#ifndef base_linear_algebra_h
#define base_linear_algebra_h


#include <deal.II/base/config.h>

#include <deal.II/base/mpi.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>

#ifdef DEAL_II_WITH_TRILINOS
#  include <deal.II/lac/trilinos_block_sparse_matrix.h>
#  include <deal.II/lac/trilinos_parallel_block_vector.h>
#  include <deal.II/lac/trilinos_precondition.h>
#  include <deal.II/lac/trilinos_solver.h>
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#  include <deal.II/lac/trilinos_sparsity_pattern.h>
#  include <deal.II/lac/trilinos_vector.h>
#endif

#ifdef DEAL_II_WITH_PETSC
#  include <deal.II/lac/petsc_block_sparse_matrix.h>
#  include <deal.II/lac/petsc_block_vector.h>
#  include <deal.II/lac/petsc_precondition.h>
#  include <deal.II/lac/petsc_solver.h>
#  include <deal.II/lac/petsc_sparse_matrix.h>
#  include <deal.II/lac/petsc_vector.h>
#endif

#include <base/partitioning.h>


template <int dim, typename MatrixType, int spacedim>
inline void
initialize_sparse_matrix(MatrixType                              &system_matrix,
                         const dealii::DoFHandler<dim, spacedim> &dof_handler,
                         const dealii::AffineConstraints<double> &constraints,
                         const Partitioning                      &partitioning)
{
  const MPI_Comm         &communicator  = dof_handler.get_communicator();
  const dealii::IndexSet &owned_dofs    = partitioning.get_owned_dofs();
  const dealii::IndexSet &relevant_dofs = partitioning.get_relevant_dofs();

  const unsigned int myid = dealii::Utilities::MPI::this_mpi_process(communicator);

  dealii::DynamicSparsityPattern dsp(relevant_dofs);

  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false, myid);

  dealii::SparsityTools::distribute_sparsity_pattern(dsp, owned_dofs, communicator, relevant_dofs);

  system_matrix.reinit(owned_dofs, owned_dofs, dsp, communicator);
}

template <int dim, typename BlockMatrixType, int spacedim>
inline void
initialize_block_sparse_matrix(BlockMatrixType                                    &system_matrix,
                               const dealii::DoFHandler<dim, spacedim>            &dof_handler,
                               const dealii::AffineConstraints<double>            &constraints,
                               const BlockPartitioning                            &block_partitioning,
                               const dealii::Table<2, dealii::DoFTools::Coupling> &coupling)
{
  const MPI_Comm                      &communicator  = dof_handler.get_communicator();
  const dealii::IndexSet              &owned_dofs    = block_partitioning.get_owned_dofs();
  const dealii::IndexSet              &relevant_dofs = block_partitioning.get_relevant_dofs();
  const std::vector<dealii::IndexSet> &owned_dofs_per_block =
    block_partitioning.get_owned_dofs_per_block();
  const std::vector<dealii::IndexSet> &relevant_dofs_per_block =
    block_partitioning.get_relevant_dofs_per_block();

  const unsigned int myid = dealii::Utilities::MPI::this_mpi_process(communicator);

  dealii::BlockDynamicSparsityPattern bdsp(relevant_dofs_per_block);

  dealii::DoFTools::make_sparsity_pattern(dof_handler, coupling, bdsp, constraints, false, myid);

  dealii::SparsityTools::distribute_sparsity_pattern(bdsp, owned_dofs, communicator, relevant_dofs);

  system_matrix.reinit(owned_dofs_per_block, owned_dofs_per_block, bdsp, communicator);
}



#ifdef DEAL_II_WITH_TRILINOS

/**
 * Convenient aliases for the linear algebra library Trilinos.
 *
 * Only available if deal.II has been configured with Trilinos.
 */
struct Trilinos
{
  using SparsityPattern = dealii::TrilinosWrappers::SparsityPattern;
  using SparseMatrix    = dealii::TrilinosWrappers::SparseMatrix;
  using Vector          = dealii::TrilinosWrappers::MPI::Vector;

  using BlockSparsityPattern = dealii::TrilinosWrappers::BlockSparsityPattern;
  using BlockSparseMatrix    = dealii::TrilinosWrappers::BlockSparseMatrix;
  using BlockVector          = dealii::TrilinosWrappers::MPI::BlockVector;

  using PreconditionAMG    = dealii::TrilinosWrappers::PreconditionAMG;
  using PreconditionJacobi = dealii::TrilinosWrappers::PreconditionJacobi;
  using SolverCG           = dealii::TrilinosWrappers::SolverCG;
};

/**
 * Convenient aliases for the linear algebra library Trilinos combined with a
 * subset of deal.II features.
 *
 * Only available if deal.II has been configured with Trilinos.
 */
struct dealiiTrilinos
{
  using SparsityPattern = dealii::TrilinosWrappers::SparsityPattern;
  using SparseMatrix    = dealii::TrilinosWrappers::SparseMatrix;
  using Vector          = dealii::LinearAlgebra::distributed::Vector<double>;

  using BlockSparsityPattern = dealii::TrilinosWrappers::BlockSparsityPattern;
  using BlockSparseMatrix    = dealii::TrilinosWrappers::BlockSparseMatrix;
  using BlockVector          = dealii::LinearAlgebra::distributed::BlockVector<double>;

  using PreconditionAMG    = dealii::TrilinosWrappers::PreconditionAMG;
  using PreconditionJacobi = dealii::TrilinosWrappers::PreconditionJacobi;
  using SolverCG           = dealii::SolverCG<Vector>;
};

template <int dim, int spacedim>
inline void
initialize_sparse_matrix(dealii::TrilinosWrappers::SparseMatrix  &system_matrix,
                         const dealii::DoFHandler<dim, spacedim> &dof_handler,
                         const dealii::AffineConstraints<double> &constraints,
                         const Partitioning                      &partitioning)
{
  const MPI_Comm         &communicator  = dof_handler.get_communicator();
  const dealii::IndexSet &owned_dofs    = partitioning.get_owned_dofs();
  const dealii::IndexSet &relevant_dofs = partitioning.get_relevant_dofs();

  const unsigned int myid = dealii::Utilities::MPI::this_mpi_process(communicator);

  dealii::TrilinosWrappers::SparsityPattern dsp(owned_dofs,
                                                owned_dofs,
                                                relevant_dofs,
                                                communicator);

  dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false, myid);

  dsp.compress();
  system_matrix.reinit(dsp);
}

template <int dim, int spacedim>
inline void
initialize_block_sparse_matrix(dealii::TrilinosWrappers::BlockSparseMatrix        &system_matrix,
                               const dealii::DoFHandler<dim, spacedim>            &dof_handler,
                               const dealii::AffineConstraints<double>            &constraints,
                               const BlockPartitioning                            &block_partitioning,
                               const dealii::Table<2, dealii::DoFTools::Coupling> &coupling)
{
  const MPI_Comm                      &communicator = dof_handler.get_communicator();
  const std::vector<dealii::IndexSet> &owned_dofs_per_block =
    block_partitioning.get_owned_dofs_per_block();
  const std::vector<dealii::IndexSet> &relevant_dofs_per_block =
    block_partitioning.get_relevant_dofs_per_block();

  const unsigned int myid = dealii::Utilities::MPI::this_mpi_process(communicator);

  dealii::TrilinosWrappers::BlockSparsityPattern bdsp(owned_dofs_per_block,
                                                      owned_dofs_per_block,
                                                      relevant_dofs_per_block,
                                                      communicator);

  dealii::DoFTools::make_sparsity_pattern(dof_handler, coupling, bdsp, constraints, false, myid);

  bdsp.compress();
  system_matrix.reinit(bdsp);
}

#else

/**
 * Convenient aliases for the linear algebra library Trilinos.
 *
 * Only available if deal.II has been configured with Trilinos.
 */
struct Trilinos
{};

/**
 * Convenient aliases for the linear algebra library Trilinos combined with a
 * subset of deal.II features.
 *
 * Only available if deal.II has been configured with Trilinos.
 */
struct dealiiTrilinos
{};

#endif // DEAL_II_WITH_TRILINOS



#ifdef DEAL_II_WITH_PETSC

/**
 * Convenient aliases for the linear algebra library PETSc.
 *
 * Only available if deal.II has been configured with PETSc.
 */
struct PETSc
{
  // petsc has no dedicated sparsitypattern class
  using SparsityPattern = dealii::DynamicSparsityPattern;
  using SparseMatrix    = dealii::PETScWrappers::MPI::SparseMatrix;
  using Vector          = dealii::PETScWrappers::MPI::Vector;

  using BlockSparsityPattern = dealii::BlockDynamicSparsityPattern;
  using BlockSparseMatrix    = dealii::PETScWrappers::MPI::BlockSparseMatrix;
  using BlockVector          = dealii::PETScWrappers::MPI::BlockVector;

  using PreconditionAMG    = dealii::PETScWrappers::PreconditionBoomerAMG;
  using PreconditionJacobi = dealii::PETScWrappers::PreconditionJacobi;
  using SolverCG           = dealii::PETScWrappers::SolverCG;
};

#else

/**
 * Convenient aliases for the linear algebra library PETSc.
 *
 * Only available if deal.II has been configured with PETSc.
 */
struct PETSc
{};

#endif // DEAL_II_WITH_PETSC


#endif
