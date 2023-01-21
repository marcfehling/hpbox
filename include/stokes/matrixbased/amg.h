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

#ifndef stokes_matrixbased_amg_h
#define stokes_matrixbased_amg_h


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <base/linear_algebra.h>
#include <base/partitioning.h>
#include <stokes/matrixbased/block_schur_preconditioner.h>


namespace StokesMatrixBased
{
  template <int dim, typename LinearAlgebra, int spacedim = dim>
  static void
  solve_amg(dealii::SolverControl                                 &solver_control_refined,
            const BlockOperatorType<dim, LinearAlgebra, spacedim> &stokes_operator,
            const OperatorType<dim, LinearAlgebra, spacedim>      &a_block_operator,
            const OperatorType<dim, LinearAlgebra, spacedim>      &schur_block_operator,
            typename LinearAlgebra::BlockVector                   &dst,
            const typename LinearAlgebra::BlockVector             &src,
            const dealii::DoFHandler<dim, spacedim>               &dof_handler)
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
    const LinearSolversMatrixBased::BlockSchurPreconditioner<LinearAlgebra,
                                                  typename LinearAlgebra::BlockSparseMatrix,
                                                  typename LinearAlgebra::SparseMatrix,
                                                  typename LinearAlgebra::SparseMatrix>
      preconditioner(stokes_operator.get_system_matrix(),
                     a_block_operator.get_system_matrix(),
                     schur_block_operator.get_system_matrix(),
                     Amg_preconditioner,
                     Mp_preconditioner,
                     true);

    // set up solver
    dealii::PrimitiveVectorMemory<typename LinearAlgebra::BlockVector> mem;

    typename dealii::SolverFGMRES<typename LinearAlgebra::BlockVector>::AdditionalData fgmres_data(
      50);
    dealii::SolverFGMRES<typename LinearAlgebra::BlockVector> solver(solver_control_refined,
                                                                     mem,
                                                                     fgmres_data);

    //
    // TODO: Is this part necessary? Or shall we just keep it matrixbased?
    //
    if constexpr (std::is_same_v<LinearAlgebra, dealiiTrilinos>)
      {
        solver.solve(stokes_operator, dst, src, preconditioner);
      }
    else
      {
        solver.solve(stokes_operator.get_system_matrix(), dst, src, preconditioner);
      }
  }
} // namespace Stokes


#endif
