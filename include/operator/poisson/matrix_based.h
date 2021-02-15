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

#ifndef operator_poisson_matrix_based_h
#define operator_poisson_matrix_based_h


#include <deal.II/base/partitioner.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/multigrid/mg_solver.h>

#include <operator/base.h>


namespace Operator
{
  namespace Poisson
  {
    template <int dim, typename VectorType>
    class MatrixBased
      : public dealii::MGSolverOperatorBase<dim,
                                            typename VectorType::value_type>
    {
    public:
      using value_type = typename VectorType::value_type;

      MatrixBased() = default;

      MatrixBased(const dealii::hp::MappingCollection<dim> &   mapping,
                  const dealii::DoFHandler<dim> &              dof_handler,
                  const dealii::hp::QCollection<dim> &         quad,
                  const dealii::AffineConstraints<value_type> &constraints,
                  VectorType &                                 system_rhs);

      void
      reinit(const dealii::hp::MappingCollection<dim> &   mapping_collection,
             const dealii::DoFHandler<dim> &              dof_handler,
             const dealii::hp::QCollection<dim> &         quadrature_collection,
             const dealii::AffineConstraints<value_type> &constraints,
             VectorType &                                 system_rhs);

      void
      vmult(VectorType &dst, const VectorType &src) const override;

      void
      initialize_dof_vector(VectorType &vec) const override;

      dealii::types::global_dof_index
      m() const override;

      void
      compute_inverse_diagonal(VectorType &diagonal) const override;

      const dealii::TrilinosWrappers::SparseMatrix &
      get_system_matrix() const override;

      void
      Tvmult(VectorType &dst, const VectorType &src) const override;

    private:
      // TODO: Add RHS function to constructor
      //       Grab and set as RHS in reinit
      // dealii::Function<dim> rhs_function;

      // TODO: Add hp::FEValues to constructor
      //       Precalculate during construction
      // dealii::hp::FEValues<dim> hp_fe_values;

      // TODO: Maybe

      dealii::TrilinosWrappers::SparseMatrix system_matrix;

      std::shared_ptr<const dealii::Utilities::MPI::Partitioner> partitioner;
    };
  } // namespace Poisson
} // namespace Operator


#endif
