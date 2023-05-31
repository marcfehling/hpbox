// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2022 by the deal.II authors
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

#ifndef poisson_amg_h
#define poisson_amg_h


#include <deal.II/lac/solver_control.h>

#include <base/linear_algebra.h>


namespace Poisson
{
  template <int dim, typename LinearAlgebra, int spacedim = dim>
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

    //
    // TODO: Is this part necessary? Or shall we just keep it matrixbased?
    //
    if constexpr (std::is_same_v<LinearAlgebra, dealiiTrilinos>)
      {
        cg.solve(poisson_operator, dst, src, preconditioner);
      }
    else
      {
        cg.solve(poisson_operator.get_system_matrix(), dst, src, preconditioner);
      }
  }
} // namespace Poisson


#endif
