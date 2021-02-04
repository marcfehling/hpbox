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
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>


namespace Solver
{
  namespace CG
  {
    class AMG
    {
    public:
      template <typename VectorType, typename Operator>
      static void
      solve(const dealii::TrilinosWrappers::PreconditionAMG::AdditionalData &data,
            dealii::SolverControl &   solver_control,
            const Operator &  system_matrix,
            VectorType &      dst,
            const VectorType &src)
      {
        dealii::TrilinosWrappers::PreconditionAMG preconditioner;
        preconditioner.initialize(system_matrix.get_system_matrix(), data);

        dealii::SolverCG<VectorType> cg(solver_control);
        cg.solve(system_matrix, dst, src, preconditioner);
      }
    };
  }
}


#endif
