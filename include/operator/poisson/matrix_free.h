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

#ifndef operator_poisson_matrixfree_h
#define operator_poisson_matrixfree_h


#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/tools.h>

#include <operator/base.h>


namespace Operator
{
  namespace Poisson
  {
    template <int dim, typename LinearAlgebra, int spacedim = dim>
    class MatrixFree : public Operator::Base<dim, LinearAlgebra, spacedim>
    {
    public:
      using VectorType = typename LinearAlgebra::Vector;
      using value_type = typename VectorType::value_type;

      using FECellIntegrator = dealii::FEEvaluation<dim, -1, 0, 1, value_type>;

      MatrixFree(
        const dealii::hp::MappingCollection<dim, spacedim> &mapping_collection,
        const dealii::hp::QCollection<dim> &quadrature_collection);

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

      const typename LinearAlgebra::SparseMatrix &
      get_system_matrix() const override;

      void
      Tvmult(VectorType &dst, const VectorType &src) const override;

    private:
      // const Parameters &prm;

      dealii::SmartPointer<const dealii::hp::MappingCollection<dim, spacedim>>
        mapping_collection;
      dealii::SmartPointer<const dealii::hp::QCollection<dim>>
        quadrature_collection;
      dealii::SmartPointer<const dealii::AffineConstraints<value_type>>
        constraints;

      // TODO: Add RHS function to constructor
      //       Grab and set as RHS in reinit
      // dealii::Function<dim> rhs_function;

      void
      do_cell_integral_local(FECellIntegrator &integrator) const;

      void
      do_cell_integral_global(FECellIntegrator &integrator,
                              VectorType &      dst,
                              const VectorType &src) const;

      void
      do_cell_integral_range(
        const dealii::MatrixFree<dim, value_type> &  matrix_free,
        VectorType &                                 dst,
        const VectorType &                           src,
        const std::pair<unsigned int, unsigned int> &range) const;

      dealii::MatrixFree<dim, value_type> matrix_free;

      mutable typename LinearAlgebra::SparseMatrix system_matrix;
    };
  } // namespace Poisson
} // namespace Operator


#endif
