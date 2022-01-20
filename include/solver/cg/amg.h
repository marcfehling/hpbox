// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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

#ifndef solver_cg_amg_h
#define solver_cg_amg_h


#include <deal.II/lac/solver_control.h>

#include <base/linear_algebra.h>


namespace Solver
{
  namespace CG
  {
    struct AMG
    {
      template <typename LinearAlgebra, typename OperatorType>
      static void
      solve(dealii::SolverControl                &solver_control,
            const OperatorType                   &system_matrix,
            typename LinearAlgebra::Vector       &dst,
            const typename LinearAlgebra::Vector &src)
      {
        typename LinearAlgebra::PreconditionAMG::AdditionalData data;
        if constexpr (std::is_same<LinearAlgebra, PETSc>::value)
          {
            data.symmetric_operator = true;
          }
        else if constexpr (std::is_same<LinearAlgebra, Trilinos>::value ||
                           std::is_same<LinearAlgebra, dealiiTrilinos>::value)
          {
            data.elliptic              = true;
            data.higher_order_elements = true;
          }
        else
          {
            Assert(false, dealii::ExcNotImplemented());
          }

        typename LinearAlgebra::PreconditionAMG preconditioner;
        preconditioner.initialize(system_matrix.get_system_matrix(), data);

        if constexpr (std::is_same<LinearAlgebra, PETSc>::value)
          {
            typename LinearAlgebra::SolverCG cg(
              solver_control,
              system_matrix.get_system_matrix().get_mpi_communicator());
            cg.solve(system_matrix.get_system_matrix(),
                     dst,
                     src,
                     preconditioner);
          }
        else if constexpr (std::is_same<LinearAlgebra, Trilinos>::value)
          {
            typename LinearAlgebra::SolverCG cg(solver_control);
            cg.solve(system_matrix.get_system_matrix(),
                     dst,
                     src,
                     preconditioner);
          }
        else
          {
            typename LinearAlgebra::SolverCG cg(solver_control);
            cg.solve(system_matrix, dst, src, preconditioner);
          }
      }
    };
  } // namespace CG
} // namespace Solver


#endif
