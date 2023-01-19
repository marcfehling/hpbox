// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the deal.II authors
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


#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <base/global.h>
#include <base/linear_algebra.h>
#include <stokes/matrixfree/schur_block_operator.h>

using namespace dealii;


namespace StokesMatrixFree
{
  template <int dim, typename LinearAlgebra, int spacedim>
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::SchurBlockOperator(
    const hp::MappingCollection<dim, spacedim> &mapping_collection,
    const hp::QCollection<dim>                 &quadrature_collection,
    const hp::FECollection<dim, spacedim>      &fe_collection)
    : mapping_collection(&mapping_collection)
    , quadrature_collection(&quadrature_collection)
  {
    (void)fe_collection; // unused, only here for matching interface to matrixbased
  }


  template <int dim, typename LinearAlgebra, int spacedim>
  std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::replicate() const
  {
    return std::make_unique<SchurBlockOperator<dim, LinearAlgebra, spacedim>>(
      *mapping_collection, *quadrature_collection, hp::FECollection<dim, spacedim>());
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::reinit(
    const Partitioning                  &partitioning,
    const DoFHandler<dim, spacedim>     &dof_handler,
    const AffineConstraints<value_type> &constraints)
  {
    Assert(false, ExcNotImplemented());
  }


  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::reinit(const Partitioning &,
                                                           const DoFHandler<dim, spacedim> &,
                                                           const AffineConstraints<value_type> &,
                                                           VectorType &,
                                                           const dealii::Function<spacedim> *)
  {
    Assert(false, ExcNotImplemented());
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::vmult(VectorType       &dst,
                                                          const VectorType &src) const
  {
    Assert(false, ExcNotImplemented());
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::initialize_dof_vector(VectorType &vec) const
  {
    Assert(false, ExcNotImplemented());
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  types::global_dof_index
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::m() const
  {
    return numbers::invalid_dof_index;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(
    VectorType &diagonal) const
  {
    Assert(false, ExcNotImplemented());
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  const typename LinearAlgebra::SparseMatrix &
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::get_system_matrix() const
  {
    Assert(false, ExcNotImplemented());
    return typename LinearAlgebra::SparseMatrix();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::Tvmult(VectorType       &dst,
                                                           const VectorType &src) const
  {
    Assert(false, ExcNotImplemented());
  }



// explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class SchurBlockOperator<2, dealiiTrilinos, 2>;
  template class SchurBlockOperator<3, dealiiTrilinos, 3>;
  template class SchurBlockOperator<2, Trilinos, 2>;
  template class SchurBlockOperator<3, Trilinos, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
  template class SchurBlockOperator<2, PETSc, 2>;
  template class SchurBlockOperator<3, PETSc, 3>;
#endif

} // namespace StokesMatrixFree
