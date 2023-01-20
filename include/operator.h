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

#ifndef operator_h
#define operator_h


#include <base/partitioning.h>
#include <mg_solver.h>

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
         const dealii::DoFHandler<dim, spacedim>     &dof_handler,
         const dealii::AffineConstraints<value_type> &constraints) = 0;

  virtual void
  reinit(const Partitioning                          &partitioning,
         const dealii::DoFHandler<dim, spacedim>     &dof_handler,
         const dealii::AffineConstraints<value_type> &constraints,
         VectorType                                  &system_rhs,
         const dealii::Function<spacedim>            *rhs_function) = 0;
  // TODO: make rhs function a reference?
};


template <int dim, typename VectorType, typename MatrixType, int spacedim = dim>
class BlockOperatorBase : public dealii::MGSolverOperatorBase<dim, VectorType, MatrixType>
{
public:
  static_assert(std::is_same_v<typename VectorType::value_type, typename MatrixType::value_type>);
  using value_type = typename VectorType::value_type;

  virtual ~BlockOperatorBase() = default;

  virtual std::unique_ptr<BlockOperatorBase<dim, VectorType, MatrixType, spacedim>>
  replicate() const = 0;

  virtual void
  reinit(const BlockPartitioning                                    &block_partitioning,
         const std::vector<dealii::DoFHandler<dim, spacedim> *>     &dof_handlers,
         const std::vector<dealii::AffineConstraints<value_type> *> &constraints) = 0;

  virtual void
  reinit(const BlockPartitioning                                    &block_partitioning,
         const std::vector<dealii::DoFHandler<dim, spacedim> *>     &dof_handlers,
         const std::vector<dealii::AffineConstraints<value_type> *> &constraints,
         VectorType                                                 &system_rhs,
         const dealii::Function<spacedim>                           *rhs_function) = 0;
  // TODO: make rhs function a reference?
}


// convenient aliases for LinearAlgebra interface
template <int dim, typename LinearAlgebra, int spacedim = dim>
using OperatorType =
  OperatorBase<dim, typename LinearAlgebra::Vector, typename LinearAlgebra::SparseMatrix, spacedim>;

template <int dim, typename LinearAlgebra, int spacedim = dim>
using BlockOperatorType = BlockOperatorBase<dim,
                                            typename LinearAlgebra::BlockVector,
                                            typename LinearAlgebra::BlockSparseMatrix,
                                            spacedim>;


#endif
