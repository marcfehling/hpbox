// ---------------------------------------------------------------------
//
// Copyright (C) 2022 - 2023 by the deal.II authors
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

#ifndef stokes_matrixbased_solvers_h
#define stokes_matrixbased_solvers_h


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <linear_algebra.h>


namespace StokesMatrixBased
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
      const bool                                        do_solve_A,
      const bool                                        do_solve_Schur_complement)
      : stokes_matrix(&stokes_matrix)
      , a_block(&a_block)
      , schur_complement_block(&schur_complement_block)
      , a_block_preconditioner(a_block_preconditioner)
      , schur_complement_preconditioner(schur_complement_preconditioner)
      , do_solve_A(do_solve_A)
      , do_solve_Schur_complement(do_solve_Schur_complement)
    {}

    void
    vmult(typename LinearAlgebra::BlockVector       &dst,
          const typename LinearAlgebra::BlockVector &src) const
    {
      // This needs to be done explicitly, as GMRES does not initialize the data of the vector dst
      // before calling us. Otherwise we might use random data as our initial guess.
      // See also: https://github.com/geodynamics/aspect/pull/4973
      dst = 0.;

      if (do_solve_Schur_complement)
        {
          dealii::SolverControl            solver_control(5000, 1e-6 * src.block(1).l2_norm());
          typename LinearAlgebra::SolverCG solver(solver_control);

          solver.solve(*schur_complement_block,
                       dst.block(1),
                       src.block(1),
                       schur_complement_preconditioner);
        }
      else
        schur_complement_preconditioner.vmult(dst.block(1), src.block(1));

      dst.block(1) *= -1.0;

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
    const bool do_solve_Schur_complement;
  };



  template <int dim, typename LinearAlgebra, int spacedim = dim>
  static void
  solve_amg(dealii::SolverControl                                 &solver_control_refined,
            const BlockOperatorType<dim, LinearAlgebra, spacedim> &stokes_operator,
            const OperatorType<dim, LinearAlgebra, spacedim>      &a_block_operator,
            const OperatorType<dim, LinearAlgebra, spacedim>      &schur_block_operator,
            typename LinearAlgebra::BlockVector                   &dst,
            const typename LinearAlgebra::BlockVector             &src,
            const dealii::DoFHandler<dim, spacedim>               &dof_handler,
            const bool                                             do_solve_A,
            const bool                                             do_solve_Schur_complement)
  {
    typename LinearAlgebra::PreconditionAMG::AdditionalData Amg_data;
    if constexpr (std::is_same<LinearAlgebra, PETSc>::value)
      {
        Amg_data.symmetric_operator = true;
        Amg_data.n_sweeps_coarse    = 2;
        Amg_data.strong_threshold   = 0.02;
      }
    else if constexpr (std::is_same<LinearAlgebra, Trilinos>::value ||
                       std::is_same<LinearAlgebra, dealiiTrilinos>::value)
      {
        std::vector<std::vector<bool>> constant_modes;

        const dealii::FEValuesExtractors::Vector velocities(0);
        dealii::DoFTools::extract_constant_modes(
          dof_handler, dof_handler.get_fe_collection().component_mask(velocities), constant_modes);

        Amg_data.constant_modes        = constant_modes;
        Amg_data.elliptic              = true;
        Amg_data.higher_order_elements = true;
        Amg_data.smoother_sweeps       = 2;
        Amg_data.aggregation_threshold = 0.02;
      }
    else
      {
        Assert(false, dealii::ExcNotImplemented());
      }

    typename LinearAlgebra::PreconditionJacobi Mp_preconditioner;
    typename LinearAlgebra::PreconditionAMG    Amg_preconditioner;

    Mp_preconditioner.initialize(schur_block_operator.get_system_matrix());
    Amg_preconditioner.initialize(a_block_operator.get_system_matrix(), Amg_data);

    //
    // TODO: System Matrix or operator? See below
    //
    const BlockSchurPreconditioner<LinearAlgebra,
                                   typename LinearAlgebra::BlockSparseMatrix,
                                   typename LinearAlgebra::SparseMatrix,
                                   typename LinearAlgebra::SparseMatrix>
      preconditioner(stokes_operator.get_system_matrix(),
                     a_block_operator.get_system_matrix(),
                     schur_block_operator.get_system_matrix(),
                     Amg_preconditioner,
                     Mp_preconditioner,
                     do_solve_A,
                     do_solve_Schur_complement);

    // set up solver
    dealii::PrimitiveVectorMemory<typename LinearAlgebra::BlockVector> mem;

    typename dealii::SolverFGMRES<typename LinearAlgebra::BlockVector>::AdditionalData fgmres_data(
      50);
    dealii::SolverFGMRES<typename LinearAlgebra::BlockVector> solver(solver_control_refined,
                                                                     mem,
                                                                     fgmres_data);

    if constexpr (std::is_same_v<LinearAlgebra, dealiiTrilinos>)
      {
        solver.solve(stokes_operator, dst, src, preconditioner);
      }
    else
      {
        solver.solve(stokes_operator.get_system_matrix(), dst, src, preconditioner);
      }
  }
} // namespace StokesMatrixBased


#endif
