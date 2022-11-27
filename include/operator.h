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


#include <mg_solver.h>

#include <memory>


// TODO: maybe use VectorType, MatrixType instead LinearAlgebra here
// in the Stokes implementation, it would be helpful when we can distinguish
// between Vector/BlockVector and SparseMatrix/BlockSparseMatrix
template <int dim, typename LinearAlgebra, int spacedim = dim>
class OperatorBase : public dealii::MGSolverOperatorBase<dim,
                                                         typename LinearAlgebra::Vector,
                                                         typename LinearAlgebra::SparseMatrix>
{
public:
  using VectorType = typename LinearAlgebra::Vector;
  using value_type = typename VectorType::value_type;

  virtual ~OperatorBase() = default;

  virtual std::unique_ptr<OperatorBase<dim, LinearAlgebra, spacedim>>
  replicate() const = 0;

  virtual void
  reinit(const dealii::DoFHandler<dim, spacedim>     &dof_handler,
         const dealii::AffineConstraints<value_type> &constraints,
         VectorType                                  &system_rhs) = 0;
};


#endif
