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


#include <deal.II/dofs/dof_tools.h>

// #include <deal.II/lac/full_matrix.h>
// #include <deal.II/lac/sparsity_tools.h>
// #include <deal.II/lac/vector.h>

#include <base/global.h>
#include <base/linear_algebra.h>
#include <operator/stokes/matrix_based.h>

using namespace dealii;


namespace Operator
{
  namespace Stokes
  {
    template <int dim, typename LinearAlgebra, int spacedim>
    MatrixBased<dim, LinearAlgebra, spacedim>::MatrixBased(
        const hp::MappingCollection<dim, spacedim> &mapping_collection,
        const hp::QCollection<dim> &                quadrature_collection,
        hp::FEValues<dim, spacedim> &               fe_values_collection)
      : mapping_collection(&mapping_collection)
      , quadrature_collection(&quadrature_collection)
      , fe_values_collection(&fe_values_collection)
    {}



    template <int dim, typename LinearAlgebra, int spacedim>
    void
    MatrixBased<dim, LinearAlgebra, spacedim>::reinit(
      const dealii::DoFHandler<dim, spacedim> &    dof_handler,
      const dealii::AffineConstraints<value_type> &constraints,
      VectorType &                                 system_rhs)
    {
      TimerOutput::Scope t(getTimer(), "reinit");

      const MPI_Comm mpi_communicator = dof_handler.get_communicator();

      // ...
    }



    template <int dim, typename LinearAlgebra, int spacedim>
    void
    MatrixBased<dim, LinearAlgebra, spacedim>::vmult(
      VectorType &      dst,
      const VectorType &src) const
    {
      system_matrix.vmult(dst, src);
    }



    template <int dim, typename LinearAlgebra, int spacedim>
    void
    MatrixBased<dim, LinearAlgebra, spacedim>::initialize_dof_vector(
      VectorType &vec) const
    {
      // ???
    }



    template <int dim, typename LinearAlgebra, int spacedim>
    types::global_dof_index
    MatrixBased<dim, LinearAlgebra, spacedim>::m() const
    {
      return system_matrix.m();
    }



    template <int dim, typename LinearAlgebra, int spacedim>
    void
    MatrixBased<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(
      VectorType &diagonal) const
    {
      this->initialize_dof_vector(diagonal);

      for (auto entry : system_matrix)
        if (entry.row() == entry.column())
          diagonal[entry.row()] = 1.0 / entry.value();
    }



    template <int dim, typename LinearAlgebra, int spacedim>
    const typename LinearAlgebra::BlockSparseMatrix &
    MatrixBased<dim, LinearAlgebra, spacedim>::get_system_matrix() const
    {
      return system_matrix;
    }



    template <int dim, typename LinearAlgebra, int spacedim>
    void
    MatrixBased<dim, LinearAlgebra, spacedim>::Tvmult(
      VectorType &      dst,
      const VectorType &src) const
    {
      vmult(dst, src);
    }



#ifdef DEAL_II_WITH_TRILINOS
    template class MatrixBased<2, dealiiTrilinos, 2>;
    template class MatrixBased<3, dealiiTrilinos, 3>;
    template class MatrixBased<2, Trilinos, 2>;
    template class MatrixBased<3, Trilinos, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
    template class MatrixBased<2, PETSc, 2>;
    template class MatrixBased<3, PETSc, 3>;
#endif

  } // namespace Stokes
} // namespace Operator
