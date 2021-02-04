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

#ifndef operator_poisson_matrix_free_h
#define operator_poisson_matrix_free_h


#include <deal.II/base/partitioner.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/tools.h>

#include <operator/base.h>


namespace Operator
{
  namespace Poisson
  {
    // A matrix-free implementation of the Laplace operator.
    template <int dim, typename VectorType>
    class MatrixFree : public Operator::Base<dim, VectorType>, public dealii::MGSolverOperatorBase<dim, typename VectorType::value_type>
    {
    public:
      using value_type = typename VectorType::value_type;

      using FECellIntegrator = dealii::FEEvaluation<dim, -1, 0, 1, value_type>;

      MatrixFree() = default;

      MatrixFree(const dealii::hp::MappingCollection<dim> &mapping,
                      const dealii::DoFHandler<dim> &           dof_handler,
                      const dealii::hp::QCollection<dim> &      quad,
                      const dealii::AffineConstraints<value_type> & constraints,
                      VectorType &                      system_rhs) override;

      void reinit(const dealii::hp::MappingCollection<dim> &mapping,
                  const dealii::DoFHandler<dim> &           dof_handler,
                  const dealii::hp::QCollection<dim> &      quad,
                  const dealii::AffineConstraints<value_type> & constraints,
                  VectorType &                      system_rhs);

      void vmult(VectorType &dst, const VectorType &src) const override;

      void initialize_dof_vector(VectorType &vec) const override;

      dealii::types::global_dof_index m() const override;

      void compute_inverse_diagonal(VectorType &diagonal) const override;

      const dealii::TrilinosWrappers::SparseMatrix &get_system_matrix() const override;

      void Tvmult(VectorType &dst, const VectorType &src) const override;

    private:
      // TODO: Add RHS function to constructor
      //       Grab and set as RHS in reinit
      // dealii::Function<dim> rhs_function;

      // TODO: Add hp::FEValues to constructor
      //       Precalculate during construction
      // dealii::hp::FEValues<dim> hp_fe_values;

      void do_cell_integral_local(FECellIntegrator &integrator) const;

      void do_cell_integral_global(FECellIntegrator &integrator,
                                   VectorType &      dst,
                                   const VectorType &src) const;

      void do_cell_integral_range(
        const dealii::MatrixFree<dim, value_type> &  matrix_free,
        VectorType &                                 dst,
        const VectorType &                           src,
        const std::pair<unsigned int, unsigned int> &range) const;

      dealii::MatrixFree<dim, value_type> matrix_free;

      dealii::AffineConstraints<value_type> constraints;

      mutable dealii::TrilinosWrappers::SparseMatrix system_matrix;
    };
  }
}


#endif
