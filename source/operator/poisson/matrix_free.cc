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


#include <deal.II/dofs/dof_tools.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <operator/poisson/matrix_free.h>

using namespace dealii;

namespace Operator
{
  namespace Poisson
  {
    template <int dim, typename VectorType>
    MatrixFree<dim, VectorType>::MatrixFree(
      const hp::MappingCollection<dim> &   mapping,
      const DoFHandler<dim> &              dof_handler,
      const hp::QCollection<dim> &         quad,
      const AffineConstraints<value_type> &constraints,
      VectorType &                         system_rhs)
    {
      this->reinit(mapping, dof_handler, quad, constraints, system_rhs);
    }



    template <int dim, typename VectorType>
    void
    MatrixFree<dim, VectorType>::reinit(
      const hp::MappingCollection<dim> &   mapping,
      const DoFHandler<dim> &              dof_handler,
      const hp::QCollection<dim> &         quad,
      const AffineConstraints<value_type> &constraints,
      VectorType &                         system_rhs)
    {
      this->system_matrix.clear();

      this->constraints.copy_from(constraints);

      typename dealii::MatrixFree<dim, value_type>::AdditionalData data;
      data.mapping_update_flags = update_gradients;

      matrix_free.reinit(mapping, dof_handler, constraints, quad, data);

      {
        AffineConstraints<value_type> constraints_without_dbc;

        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                locally_relevant_dofs);
        constraints_without_dbc.reinit(locally_relevant_dofs);

        DoFTools::make_hanging_node_constraints(dof_handler,
                                                constraints_without_dbc);
        constraints_without_dbc.close();

        VectorType b, x;

        this->initialize_dof_vector(system_rhs);
        this->initialize_dof_vector(b);
        this->initialize_dof_vector(x);

        dealii::MatrixFree<dim, value_type> matrix_free;
        matrix_free.reinit(
          mapping, dof_handler, constraints_without_dbc, quad, data);

        constraints.distribute(x);

        matrix_free.cell_loop(&MatrixFree::do_cell_integral_range, this, b, x);

        constraints.set_zero(b);

        system_rhs -= b;
      }
    }



    template <int dim, typename VectorType>
    void
    MatrixFree<dim, VectorType>::vmult(VectorType &      dst,
                                       const VectorType &src) const
    {
      this->matrix_free.cell_loop(
        &MatrixFree::do_cell_integral_range, this, dst, src, true);
    }



    template <int dim, typename VectorType>
    void
    MatrixFree<dim, VectorType>::initialize_dof_vector(VectorType &vec) const
    {
      matrix_free.initialize_dof_vector(vec);
    }



    template <int dim, typename VectorType>
    types::global_dof_index
    MatrixFree<dim, VectorType>::m() const
    {
      return matrix_free.get_dof_handler().n_dofs();
    }



    template <int dim, typename VectorType>
    void
    MatrixFree<dim, VectorType>::compute_inverse_diagonal(
      VectorType &diagonal) const
    {
      MatrixFreeTools::compute_diagonal(matrix_free,
                                        diagonal,
                                        &MatrixFree::do_cell_integral_local,
                                        this);

      // invert diagonal
      for (auto &i : diagonal)
        i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
    }



    template <int dim, typename VectorType>
    const TrilinosWrappers::SparseMatrix &
    MatrixFree<dim, VectorType>::get_system_matrix() const
    {
      // Check if matrix has already been set up.
      if (system_matrix.m() == 0 && system_matrix.n() == 0)
        {
          const auto &dof_handler = this->matrix_free.get_dof_handler();
          TrilinosWrappers::SparsityPattern dsp(
            dof_handler.locally_owned_dofs(), get_mpi_comm(dof_handler));
          DoFTools::make_sparsity_pattern(dof_handler, dsp, this->constraints);

          dsp.compress();
          system_matrix.reinit(dsp);

          MatrixFreeTools::compute_matrix(matrix_free,
                                          constraints,
                                          system_matrix,
                                          &MatrixFree::do_cell_integral_local,
                                          this);
        }

      return this->system_matrix;
    }



    template <int dim, typename VectorType>
    void
    MatrixFree<dim, VectorType>::Tvmult(VectorType &      dst,
                                        const VectorType &src) const
    {
      vmult(dst, src);
    }



    template <int dim, typename VectorType>
    void
    MatrixFree<dim, VectorType>::do_cell_integral_local(
      FECellIntegrator &integrator) const
    {
      integrator.evaluate(EvaluationFlags::gradients);

      for (unsigned int q = 0; q < integrator.n_q_points; ++q)
        integrator.submit_gradient(integrator.get_gradient(q), q);

      integrator.integrate(EvaluationFlags::gradients);
    }



    template <int dim, typename VectorType>
    void
    MatrixFree<dim, VectorType>::do_cell_integral_global(
      FECellIntegrator &integrator,
      VectorType &      dst,
      const VectorType &src) const
    {
      integrator.gather_evaluate(src, EvaluationFlags::gradients);

      for (unsigned int q = 0; q < integrator.n_q_points; ++q)
        integrator.submit_gradient(integrator.get_gradient(q), q);

      integrator.integrate_scatter(EvaluationFlags::gradients, dst);
    }



    template <int dim, typename VectorType>
    void
    MatrixFree<dim, VectorType>::do_cell_integral_range(
      const dealii::MatrixFree<dim, value_type> &  matrix_free,
      VectorType &                                 dst,
      const VectorType &                           src,
      const std::pair<unsigned int, unsigned int> &range) const
    {
      FECellIntegrator integrator(matrix_free, range);

      for (unsigned cell = range.first; cell < range.second; ++cell)
        {
          integrator.reinit(cell);

          do_cell_integral_global(integrator, dst, src);
        }
    }



    // explicit instantiations
    using VectorType = LinearAlgebra::distributed::Vector<double>;
    template class MatrixFree<2, VectorType>;
    template class MatrixFree<3, VectorType>;
  } // namespace Poisson
} // namespace Operator
