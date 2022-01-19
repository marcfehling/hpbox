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

#ifndef operator_stokes_matrixbased_h
#define operator_stokes_matrixbased_h


#include <deal.II/base/partitioner.h>

#include <deal.II/hp/fe_values.h>

#include <operator/base.h>


namespace Operator
{
  namespace Stokes
  {
    template <int dim, typename LinearAlgebra, int spacedim = dim>
    class MatrixBased : public Operator::BaseBlock<dim, LinearAlgebra, spacedim>
    {
      using VectorType = typename LinearAlgebra::BlockVector;
      using value_type = typename VectorType::value_type;

      MatrixBased(
        const dealii::hp::MappingCollection<dim, spacedim> &mapping_collection,
        const dealii::hp::QCollection<dim> & quadrature_collection,
        dealii::hp::FEValues<dim, spacedim> &fe_values_collection);

      void
      reinit(const dealii::DoFHandler<dim, spacedim> &    dof_handler,
             const dealii::AffineConstraints<value_type> &constraints,
             VectorType &                                 system_rhs) override;

      void
      vmult(VectorType &dst, const VectorType &src) const override;

      void
      initialize_dof_vector(VectorType &vec) const override;

      dealii::types::global_dof_index
      m() const override;

      void
      compute_inverse_diagonal(VectorType &diagonal) const override;

      const typename LinearAlgebra::BlockSparseMatrix &
      get_system_matrix() const override;

      void
      Tvmult(VectorType &dst, const VectorType &src) const override;

    private:
      // const Parameters &prm;

      dealii::SmartPointer<const dealii::hp::MappingCollection<dim, spacedim>>
        mapping_collection;
      dealii::SmartPointer<const dealii::hp::QCollection<dim>>
        quadrature_collection;

      // TODO: Add RHS function to constructor
      //       Grab and set as RHS in reinit
      // dealii::Function<dim> rhs_function;

      dealii::SmartPointer<dealii::hp::FEValues<dim, spacedim>>
        fe_values_collection;

      typename LinearAlgebra::BlockSparseMatrix system_matrix;

      std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner;
    };
  } // namespace Stokes
} // namespace Operator


#endif
