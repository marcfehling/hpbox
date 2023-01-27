// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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

#ifndef stokes_matrixfree_block_schur_preconditioner_h
#define stokes_matrixfree_block_schur_preconditioner_h


#include <base/linear_algebra.h>


namespace LinearSolversMatrixFree
{
  template <typename LinearAlgebra,
            typename StokesMatrixType,
            typename ABlockMatrixType,
            typename SchurComplementMatrixType>
  class BlockSchurPreconditioner : public dealii::Subscriptor
  {
  public:
    BlockSchurPreconditioner(
      const StokesMatrixType                           &stokes_matrix,
      const ABlockMatrixType                           &a_block,
      const SchurComplementMatrixType                  &schur_complement_block,
      const typename LinearAlgebra::PreconditionAMG    &a_block_preconditioner,
      const typename LinearAlgebra::PreconditionJacobi &schur_complement_preconditioner,
      const bool                                        do_solve_A)
      : stokes_matrix(&stokes_matrix)
      , a_block(&a_block)
      , schur_complement_block(&schur_complement_block)
      , a_block_preconditioner(a_block_preconditioner)
      , schur_complement_preconditioner(schur_complement_preconditioner)
      , do_solve_A(do_solve_A)
    {}

    void
    vmult(typename LinearAlgebra::BlockVector       &dst,
          const typename LinearAlgebra::BlockVector &src) const
    {
      // This needs to be done explicitly, as GMRES does not initialize the data of the vector dst
      // before calling us. Otherwise we might use random data as our initial guess.
      // See also: https://github.com/geodynamics/aspect/pull/4973
      dst = 0.;

      {
        dealii::SolverControl            solver_control(5000, 1e-6 * src.block(1).l2_norm());
        typename LinearAlgebra::SolverCG solver(solver_control);

        solver.solve(*schur_complement_block,
                     dst.block(1),
                     src.block(1),
                     schur_complement_preconditioner);

        dst.block(1) *= -1.0;
      }

      typename LinearAlgebra::Vector utmp(src.block(0));

      {
        stokes_matrix->block(0, 1).vmult(utmp, dst.block(1));
        utmp *= -1.0;
        utmp += src.block(0);
      }

      if (do_solve_A == true)
        {
          dealii::SolverControl            solver_control(5000, 1e-2 * utmp.l2_norm());
          typename LinearAlgebra::SolverCG solver(solver_control);

          solver.solve(*a_block, dst.block(0), utmp, a_block_preconditioner);
        }
      else
        a_block_preconditioner.vmult(dst.block(0), utmp);
    }

  private:
    const dealii::SmartPointer<const StokesMatrixType>          stokes_matrix;
    const dealii::SmartPointer<const ABlockMatrixType>          a_block;
    const dealii::SmartPointer<const SchurComplementMatrixType> schur_complement_block;

    const typename LinearAlgebra::PreconditionAMG    &a_block_preconditioner;
    const typename LinearAlgebra::PreconditionJacobi &schur_complement_preconditioner;

    const bool do_solve_A;
  };
} // namespace LinearSolversMatrixFree


#endif
