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

#ifndef multigrid_operator_base_h
#define multigrid_operator_base_h


#include <deal.II/base/subscriptor.h>
#include <deal.II/base/types.h>


// ----------------------------------------
// NOTE:
// This part is modified from WIP PR #11699 which determines the interface
// of the MGSolverOperatorBase class.


DEAL_II_NAMESPACE_OPEN

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



template <int dim, typename VectorType, typename MatrixType>
types::global_dof_index
MGSolverOperatorBase<dim, VectorType, MatrixType>::m() const
{
  Assert(false, ExcNotImplemented());
  return 0;
}



template <int dim, typename VectorType, typename MatrixType>
typename MGSolverOperatorBase<dim, VectorType, MatrixType>::value_type
MGSolverOperatorBase<dim, VectorType, MatrixType>::el(unsigned int, unsigned int) const
{
  Assert(false, ExcNotImplemented());
  return 0;
}



template <int dim, typename VectorType, typename MatrixType>
void
MGSolverOperatorBase<dim, VectorType, MatrixType>::initialize_dof_vector(VectorType &vec) const
{
  Assert(false, ExcNotImplemented());
  (void)vec;
}



template <int dim, typename VectorType, typename MatrixType>
void
MGSolverOperatorBase<dim, VectorType, MatrixType>::vmult(VectorType       &dst,
                                                         const VectorType &src) const
{
  Assert(false, ExcNotImplemented());
  (void)dst;
  (void)src;
}



template <int dim, typename VectorType, typename MatrixType>
void
MGSolverOperatorBase<dim, VectorType, MatrixType>::Tvmult(VectorType       &dst,
                                                          const VectorType &src) const
{
  Assert(false, ExcNotImplemented());
  (void)dst;
  (void)src;
}



template <int dim, typename VectorType, typename MatrixType>
void
MGSolverOperatorBase<dim, VectorType, MatrixType>::compute_inverse_diagonal(
  VectorType &diagonal) const
{
  Assert(false, ExcNotImplemented());
  (void)diagonal;
}



template <int dim, typename VectorType, typename MatrixType>
const MatrixType &
MGSolverOperatorBase<dim, VectorType, MatrixType>::get_system_matrix() const
{
  Assert(false, ExcNotImplemented());
  return dummy_sparse_matrix;
}

DEAL_II_NAMESPACE_CLOSE


// ----------------------------------------


#include <deal.II/matrix_free/matrix_free.h>

#include <multigrid/mg_solver.h>
#include <partitioning.h>

#include <memory>


template <int dim, typename VectorType, typename MatrixType, int spacedim = dim>
class OperatorBase : public dealii::MGSolverOperatorBase<dim, VectorType, MatrixType>
{
public:
  static_assert(std::is_same_v<typename VectorType::value_type, typename MatrixType::value_type>);
  using value_type = typename VectorType::value_type;

  virtual ~OperatorBase() = default;

  virtual std::unique_ptr<OperatorBase<dim, VectorType, MatrixType, spacedim>>
  replicate() const = 0;

  virtual void
  reinit(const Partitioning                          &partitioning,
         const dealii::MatrixFree<dim, value_type>   &matrix_free,
         const dealii::AffineConstraints<value_type> &constraints) = 0;

  virtual void
  reinit(const Partitioning                          &partitioning,
         const dealii::DoFHandler<dim, spacedim>     &dof_handler,
         const dealii::AffineConstraints<value_type> &constraints) = 0;

  virtual void
  reinit(const Partitioning                          &partitioning,
         const dealii::DoFHandler<dim, spacedim>     &dof_handler,
         const dealii::AffineConstraints<value_type> &constraints,
         VectorType                                  &system_rhs,
         const dealii::Function<spacedim>            *rhs_function) = 0;
  // TODO: make rhs function a reference?

  virtual const dealii::MatrixFree<dim, value_type> &
  get_matrix_free() const = 0;

  virtual const MatrixType &
  get_system_matrix() const override = 0;
};


// convenient aliases for LinearAlgebra interface
template <int dim, typename LinearAlgebra, int spacedim = dim>
using OperatorType =
  OperatorBase<dim, typename LinearAlgebra::Vector, typename LinearAlgebra::SparseMatrix, spacedim>;

template <int dim, typename LinearAlgebra, int spacedim = dim>
using BlockOperatorType = OperatorBase<dim,
                                       typename LinearAlgebra::BlockVector,
                                       typename LinearAlgebra::BlockSparseMatrix,
                                       spacedim>;


#endif
