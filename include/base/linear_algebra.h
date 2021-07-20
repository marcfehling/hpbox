// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
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

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>

#ifdef DEAL_II_WITH_TRILINOS
#  include <deal.II/lac/trilinos_precondition.h>
#  include <deal.II/lac/trilinos_solver.h>
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#  include <deal.II/lac/trilinos_sparsity_pattern.h>
#  include <deal.II/lac/trilinos_vector.h>
#endif

#ifdef DEAL_II_WITH_PETSC
#  include <deal.II/lac/petsc_precondition.h>
#  include <deal.II/lac/petsc_solver.h>
#  include <deal.II/lac/petsc_sparse_matrix.h>
#  include <deal.II/lac/petsc_vector.h>
#endif



#ifdef DEAL_II_WITH_TRILINOS

/**
 * Convenient aliases for the linear algebra library Trilinos.
 *
 * Only available if deal.II has been configured with Trilinos.
 */
struct Trilinos
{
  using SparsityPattern = dealii::TrilinosWrappers::SparsityPattern;

  using SparseMatrix = dealii::TrilinosWrappers::SparseMatrix;

  using Vector = dealii::TrilinosWrappers::MPI::Vector;

  using PreconditionAMG = dealii::TrilinosWrappers::PreconditionAMG;

  using SolverCG = dealii::TrilinosWrappers::SolverCG;
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

  using SparseMatrix = dealii::TrilinosWrappers::SparseMatrix;

  using Vector = dealii::LinearAlgebra::distributed::Vector<double>;

  using PreconditionAMG = dealii::TrilinosWrappers::PreconditionAMG;

  using SolverCG = dealii::SolverCG<Vector>;
};

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

  using SparseMatrix = dealii::PETScWrappers::MPI::SparseMatrix;

  using Vector = dealii::PETScWrappers::MPI::Vector;

  using PreconditionAMG = dealii::PETScWrappers::PreconditionBoomerAMG;

  using SolverCG = dealii::PETScWrappers::SolverCG;
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
