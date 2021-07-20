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

#ifndef operator_factory_h
#define operator_factory_h


#include <deal.II/base/exceptions.h>

#include <operator/poisson/matrix_based.h>
#include <operator/poisson/matrix_free.h>


namespace Factory
{
  template <int dim,
            typename LinearAlgebra,
            int spacedim = dim,
            typename... Args>
  std::unique_ptr<
    dealii::MGSolverOperatorBase<dim,
                                 typename LinearAlgebra::Vector,
                                 typename LinearAlgebra::SparseMatrix>>
  create_operator(const std::string &type, Args &&...args)
  {
    if (type == "poisson")
      return std::make_unique<
        Operator::Poisson::MatrixBased<dim, LinearAlgebra, spacedim>>(
        std::forward<Args>(args)...);
    else if (type == "poisson")
      return std::make_unique<
        Operator::Poisson::MatrixFree<dim, LinearAlgebra, spacedim>>(
        std::forward<Args>(args)...);

    Assert(false, dealii::ExcNotImplemented());
    return std::unique_ptr<
      dealii::MGSolverOperatorBase<dim,
                                   typename LinearAlgebra::Vector,
                                   typename LinearAlgebra::SparseMatrix>>();
  }
} // namespace Factory


#endif
