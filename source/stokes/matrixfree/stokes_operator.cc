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


#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <base/global.h>
#include <base/linear_algebra.h>
#include <stokes/matrixfree/stokes_operator.h>

using namespace dealii;


namespace StokesMatrixFree
{
  template <int dim, typename LinearAlgebra, int spacedim>
  StokesOperator<dim, LinearAlgebra, spacedim>::StokesOperator(
    const hp::MappingCollection<dim, spacedim> &mapping_collection,
    const std::vector<hp::QCollection<dim>>    &quadrature_collections)
    : mapping_collection(&mapping_collection)
    , quadrature_collections(&quadrature_collections)
  {}



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::reinit(
    const std::vector<const Partitioning *>                  &partitionings,
    const std::vector<const DoFHandler<dim, spacedim> *>     &dof_handlers,
    const std::vector<const AffineConstraints<value_type> *> &constraints,
    VectorType                                               &system_rhs,
    const std::vector<const dealii::Function<spacedim> *>    &rhs_functions)
  {
    TimerOutput::Scope t(getTimer(), "reinit");

    this->constraints = &constraints;
    this->rhs_functions = &rhs_functions;

    typename MatrixFree<dim, value_type>::AdditionalData data;
    data.mapping_update_flags = update_gradients;
    // TODO: more?

    matrix_free.reinit(*mapping_collection, dof_handlers, constraints, *quadrature_collections, data);

    this->initialize_dof_vector(system_rhs);

    // residual: r = f - Au0
    // TODO: that is just the -Au0 part. add the rhs function part (check step-37/step-67)
    {
      AffineConstraints<value_type> constraints_v_without_dbc;
      constraints_v_without_dbc.reinit(partitionings[0]->get_relevant_dofs());
      DoFTools::make_hanging_node_constraints(*dof_handlers[0], constraints_v_without_dbc);
      constraints_v_without_dbc.close();

      AffineConstraints<value_type> constraints_p_without_dbc;
      constraints_p_without_dbc.reinit(partitionings[1]->get_relevant_dofs());
      DoFTools::make_hanging_node_constraints(*dof_handlers[1], constraints_p_without_dbc);
      constraints_p_without_dbc.close();

      const std::vector<const AffineConstraints<value_type> *> constraints_without_dbc = {&constraints_v_without_dbc, &constraints_p_without_dbc};

      MatrixFree<dim, value_type> matrix_free;
      matrix_free.reinit(
        *mapping_collection, dof_handlers, constraints_without_dbc, *quadrature_collections, data);

      VectorType b;
      b.reinit(2);
      matrix_free.initialize_dof_vector(b.block(0), 0);
      matrix_free.initialize_dof_vector(b.block(1), 1);
      b.collect_sizes();

      VectorType x;
      x.reinit(2);
      matrix_free.initialize_dof_vector(x.block(0), 0);
      matrix_free.initialize_dof_vector(x.block(1), 1);
      x.collect_sizes();

      constraints[0]->distribute(x.block(0));
      constraints[1]->distribute(x.block(1));

      // only zero rhs function supported for now
      // TODO: evaluate rhs function here

      matrix_free.cell_loop(&StokesOperator::do_cell_integral_range, this, b, x);

      constraints[0]->set_zero(b.block(0));
      constraints[1]->set_zero(b.block(1));

      system_rhs -= b;
    }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::vmult(VectorType &dst, const VectorType &src) const
  {
    TimerOutput::Scope t(getTimer(), "vmult");

    this->matrix_free.cell_loop(&StokesOperator::do_cell_integral_range, this, dst, src, true);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::initialize_dof_vector(VectorType &vec) const
  {
    vec.reinit(2);
    matrix_free.initialize_dof_vector(vec.block(0), 0);
    matrix_free.initialize_dof_vector(vec.block(1), 1);
    vec.collect_sizes();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  types::global_dof_index
  StokesOperator<dim, LinearAlgebra, spacedim>::m() const
  {
    return matrix_free.get_dof_handler(0).n_dofs() + matrix_free.get_dof_handler(1).n_dofs();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(
    VectorType & /*diagonal*/) const
  {
    Assert(false, ExcNotImplemented());
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  const typename LinearAlgebra::BlockSparseMatrix &
  StokesOperator<dim, LinearAlgebra, spacedim>::get_system_matrix() const
  {
    Assert(false, ExcNotImplemented());
    return dummy;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::Tvmult(VectorType &dst, const VectorType &src) const
  {
    this->vmult(dst, src);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::do_cell_integral_range(
    const MatrixFree<dim, value_type>           &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim, -1, 0, dim, value_type> velocity (matrix_free, range, 0);
    FEEvaluation<dim, -1, 0, 1  , value_type> pressure (matrix_free, range, 1);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        velocity.reinit (cell);
        velocity.gather_evaluate (src.block(0), EvaluationFlags::gradients);
        pressure.reinit (cell);
        pressure.gather_evaluate (src.block(1), EvaluationFlags::values);

        for (unsigned int q = 0; q < velocity.n_q_points; ++q)
          {
            Tensor<1, dim, Tensor<1, dim, VectorizedArray<double>>> grad_u =
              velocity.get_gradient (q);
            VectorizedArray<double> pres = pressure.get_value(q);
            VectorizedArray<double> div_u = velocity.get_divergence (q);
            pressure.submit_value (-div_u, q);

            // TODO: Move viscosity to class member
            constexpr double viscosity = 0.1;
            grad_u *= viscosity;

            // subtract p * I
            for (unsigned int d=0; d<dim; ++d)
              grad_u[d][d] -= pres;

            velocity.submit_gradient(grad_u, q);
         }

        velocity.integrate_scatter (EvaluationFlags::gradients, dst.block(0));
        pressure.integrate_scatter (EvaluationFlags::values, dst.block(1));
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::do_cell_rhs_function_range(
      const MatrixFree<dim, value_type>           &matrix_free,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim, -1, 0, dim, value_type> velocity (matrix_free, range, 0);
    FEEvaluation<dim, -1, 0, 1  , value_type> pressure (matrix_free, range, 1);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        velocity.reinit (cell);
        velocity.read_dof_values (src.block(0));
        velocity.evaluate (EvaluationFlags::values | EvaluationFlags::gradients);
        pressure.reinit (cell);
        pressure.read_dof_values (src.block(1));
        pressure.evaluate (EvaluationFlags::values);

        for (unsigned int q = 0; q < velocity.n_q_points; ++q)
          {
            // do something like step-67
         }

        velocity.integrate (EvaluationFlags::values | EvaluationFlags::gradients);
        velocity.distribute_local_to_global (dst.block(0));
        pressure.integrate (EvaluationFlags::values);
        pressure.distribute_local_to_global (dst.block(1));
      }
  }



// explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class StokesOperator<2, dealiiTrilinos, 2>;
  template class StokesOperator<3, dealiiTrilinos, 3>;
#endif

} // namespace StokesMatrixFree
