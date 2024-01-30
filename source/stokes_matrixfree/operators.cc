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

#include <global.h>
#include <linear_algebra.h>
#include <stokes_matrixfree/operators.h>

using namespace dealii;


namespace StokesMatrixFree
{
  // --- ABlockOperator ---
  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::reinit(
    const Partitioning                                 &partitioning,
    const std::shared_ptr<MatrixFree<dim, value_type>> &matrix_free,
    const AffineConstraints<value_type>                &constraints)
  {
    TimerOutput::Scope t(getTimer(), "reinit_ABlockOperator");

    this->a_block_matrix.clear();

    this->partitioning = partitioning;
    this->constraints  = &constraints;

    this->matrix_free = matrix_free;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::reinit(
    const Partitioning                         &partitioning,
    const hp::MappingCollection<dim, spacedim> &mapping_collection,
    const DoFHandler<dim, spacedim>            &dof_handler,
    const AffineConstraints<value_type>        &constraints,
    const hp::QCollection<dim>                 &quadrature_collection)
  {
    TimerOutput::Scope t(getTimer(), "reinit_ABlockOperator");

    this->a_block_matrix.clear();

    this->partitioning = partitioning;
    this->constraints  = &constraints;

    typename MatrixFree<dim, value_type>::AdditionalData data;
    data.mapping_update_flags = update_gradients;

    this->matrix_free = std::make_shared<MatrixFree<dim, value_type>>();
    this->matrix_free->reinit(
      mapping_collection, dof_handler, constraints, quadrature_collection, data);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::vmult(VectorType &dst, const VectorType &src) const
  {
    TimerOutput::Scope t(getTimer(), "vmult_ABlockOperator");

    matrix_free->cell_loop(&ABlockOperator::do_cell_integral_range, this, dst, src, true);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::initialize_dof_vector(VectorType &vec) const
  {
    matrix_free->initialize_dof_vector(vec, velocity_index);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  types::global_dof_index
  ABlockOperator<dim, LinearAlgebra, spacedim>::m() const
  {
    return matrix_free->get_dof_handler(velocity_index).n_dofs();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(VectorType &diagonal) const
  {
    initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(
      *matrix_free, diagonal, &ABlockOperator::do_cell_integral_local, this, velocity_index);

    // invert diagonal
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  const typename LinearAlgebra::SparseMatrix &
  ABlockOperator<dim, LinearAlgebra, spacedim>::get_system_matrix() const
  {
    // Check if matrix has already been set up.
    if (a_block_matrix.m() == 0 && a_block_matrix.n() == 0)
      {
        initialize_sparse_matrix(a_block_matrix,
                                 matrix_free->get_dof_handler(velocity_index),
                                 *constraints,
                                 partitioning);

        MatrixFreeTools::compute_matrix(*matrix_free,
                                        *constraints,
                                        a_block_matrix,
                                        &ABlockOperator::do_cell_integral_local,
                                        this,
                                        velocity_index);
      }

    return a_block_matrix;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::Tvmult(VectorType &dst, const VectorType &src) const
  {
    this->vmult(dst, src);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::do_cell_integral_local(
    FECellIntegrator &velocity) const
  {
    velocity.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < velocity.n_q_points; ++q)
      {
        Tensor<1, dim, Tensor<1, dim, VectorizedArray<double>>> grad_u = velocity.get_gradient(q);

        // TODO: Move viscosity to class member
        constexpr double viscosity = 0.1;
        grad_u *= viscosity;

        velocity.submit_gradient(grad_u, q);
      }

    velocity.integrate(EvaluationFlags::gradients);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::do_cell_integral_global(FECellIntegrator &velocity,
                                                                        VectorType       &dst,
                                                                        const VectorType &src) const
  {
    velocity.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < velocity.n_q_points; ++q)
      {
        Tensor<1, dim, Tensor<1, dim, VectorizedArray<double>>> grad_u = velocity.get_gradient(q);

        // TODO: Move viscosity to class member
        constexpr double viscosity = 0.1;
        grad_u *= viscosity;

        velocity.submit_gradient(grad_u, q);
      }

    velocity.integrate_scatter(EvaluationFlags::gradients, dst);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::do_cell_integral_range(
    const MatrixFree<dim, value_type>           &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator velocity(matrix_free, range, velocity_index);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        velocity.reinit(cell);

        do_cell_integral_global(velocity, dst, src);
      }
  }



  // --- SchurBlockOperator ---
  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::reinit(
    const Partitioning                                 &partitioning,
    const std::shared_ptr<MatrixFree<dim, value_type>> &matrix_free,
    const AffineConstraints<value_type>                &constraints)
  {
    TimerOutput::Scope t(getTimer(), "reinit_SchurBlockOperator");

    this->schur_block_matrix.clear();

    this->partitioning = partitioning;
    this->constraints  = &constraints;

    this->matrix_free = matrix_free;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::vmult(VectorType       &dst,
                                                          const VectorType &src) const
  {
    TimerOutput::Scope t(getTimer(), "vmult_SchurBlockOperator");

    matrix_free->cell_loop(&SchurBlockOperator::do_cell_integral_range, this, dst, src, true);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::initialize_dof_vector(VectorType &vec) const
  {
    matrix_free->initialize_dof_vector(vec, pressure_index);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  types::global_dof_index
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::m() const
  {
    return matrix_free->get_dof_handler().n_dofs();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(
    VectorType &diagonal) const
  {
    initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(
      *matrix_free, diagonal, &SchurBlockOperator::do_cell_integral_local, this, pressure_index);

    // invert diagonal
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  const typename LinearAlgebra::SparseMatrix &
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::get_system_matrix() const
  {
    // Check if matrix has already been set up.
    if (schur_block_matrix.m() == 0 && schur_block_matrix.n() == 0)
      {
        initialize_sparse_matrix(schur_block_matrix,
                                 matrix_free->get_dof_handler(pressure_index),
                                 *constraints,
                                 partitioning);

        MatrixFreeTools::compute_matrix(*matrix_free,
                                        *constraints,
                                        schur_block_matrix,
                                        &SchurBlockOperator::do_cell_integral_local,
                                        this,
                                        pressure_index);
      }

    return schur_block_matrix;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::Tvmult(VectorType       &dst,
                                                           const VectorType &src) const
  {
    this->vmult(dst, src);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::do_cell_integral_local(
    FECellIntegrator &pressure) const
  {
    pressure.evaluate(EvaluationFlags::values);

    for (unsigned int q = 0; q < pressure.n_q_points; ++q)
      {
        VectorizedArray<double> value = pressure.get_value(q);

        // TODO: move to class member
        constexpr double viscosity     = 0.1;
        constexpr double inv_viscosity = 1 / viscosity;
        value *= inv_viscosity;

        pressure.submit_value(value, q);
      }

    pressure.integrate(EvaluationFlags::values);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::do_cell_integral_global(
    FECellIntegrator &pressure,
    VectorType       &dst,
    const VectorType &src) const
  {
    pressure.gather_evaluate(src, EvaluationFlags::values);

    for (unsigned int q = 0; q < pressure.n_q_points; ++q)
      {
        VectorizedArray<double> value = pressure.get_value(q);

        // TODO: move to class member
        constexpr double viscosity     = 0.1;
        constexpr double inv_viscosity = 1 / viscosity;
        value *= inv_viscosity;

        pressure.submit_value(value, q);
      }

    pressure.integrate_scatter(EvaluationFlags::values, dst);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::do_cell_integral_range(
    const MatrixFree<dim, value_type>           &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator pressure(matrix_free, range, pressure_index);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        pressure.reinit(cell);

        do_cell_integral_global(pressure, dst, src);
      }
  }



  // --- StokesBlockOperator ---
  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::reinit(
    const std::vector<const Partitioning *>                  &partitionings,
    const hp::MappingCollection<dim, spacedim>               &mapping_collection,
    const std::vector<const DoFHandler<dim, spacedim> *>     &dof_handlers,
    const std::vector<const AffineConstraints<value_type> *> &constraints,
    const hp::QCollection<dim>                               &quadrature_collection,
    VectorType                                               &system_rhs,
    const std::vector<const dealii::Function<spacedim> *>    &rhs_functions)
  {
    TimerOutput::Scope t(getTimer(), "reinit_StokesOperator");

    typename MatrixFree<dim, value_type>::AdditionalData data;
    data.mapping_update_flags = update_gradients | update_quadrature_points;
    // TODO: we need quad points only for rhs function. hide between nullptr check

    this->matrix_free = std::make_shared<MatrixFree<dim, value_type>>();
    this->matrix_free->reinit(
      mapping_collection, dof_handlers, constraints, quadrature_collection, data);

    this->rhs_functions = rhs_functions;
    this->initialize_dof_vector(system_rhs);
    // TODO: check if nullptr
    this->matrix_free->cell_loop(&StokesOperator::do_cell_rhs_function_range,
                                 this,
                                 system_rhs,
                                 system_rhs);

    // residual: r = f - Au0
    // TODO: that is just the -Au0 part. add the rhs function part (check step-37/step-67)
    {
      AffineConstraints<value_type> constraints_v_without_dbc;
      constraints_v_without_dbc.reinit(partitionings[velocity_index]->get_relevant_dofs());
      DoFTools::make_hanging_node_constraints(*dof_handlers[velocity_index],
                                              constraints_v_without_dbc);
      constraints_v_without_dbc.close();

      AffineConstraints<value_type> constraints_p_without_dbc;
      constraints_p_without_dbc.reinit(partitionings[pressure_index]->get_relevant_dofs());
      DoFTools::make_hanging_node_constraints(*dof_handlers[pressure_index],
                                              constraints_p_without_dbc);
      constraints_p_without_dbc.close();

      const std::vector<const AffineConstraints<value_type> *> constraints_without_dbc = {
        &constraints_v_without_dbc, &constraints_p_without_dbc};

      MatrixFree<dim, value_type> matrix_free;
      matrix_free.reinit(
        mapping_collection, dof_handlers, constraints_without_dbc, quadrature_collection, data);

      VectorType b;
      b.reinit(2);
      matrix_free.initialize_dof_vector(b.block(velocity_index), velocity_index);
      matrix_free.initialize_dof_vector(b.block(pressure_index), pressure_index);
      b.collect_sizes();

      VectorType x;
      x.reinit(2);
      matrix_free.initialize_dof_vector(x.block(velocity_index), velocity_index);
      matrix_free.initialize_dof_vector(x.block(pressure_index), pressure_index);
      x.collect_sizes();

      constraints[velocity_index]->distribute(x.block(velocity_index));
      constraints[pressure_index]->distribute(x.block(pressure_index));

      // only zero rhs function supported for now
      // TODO: evaluate rhs function here

      matrix_free.cell_loop(&StokesOperator::do_cell_integral_range, this, b, x);

      constraints[velocity_index]->set_zero(b.block(velocity_index));
      constraints[pressure_index]->set_zero(b.block(pressure_index));

      system_rhs -= b;
    }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::vmult(VectorType &dst, const VectorType &src) const
  {
    TimerOutput::Scope t(getTimer(), "vmult_StokesOperator");

    matrix_free->cell_loop(&StokesOperator::do_cell_integral_range, this, dst, src, true);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::initialize_dof_vector(VectorType &vec) const
  {
    vec.reinit(2);
    matrix_free->initialize_dof_vector(vec.block(velocity_index), velocity_index);
    matrix_free->initialize_dof_vector(vec.block(pressure_index), pressure_index);
    vec.collect_sizes();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  types::global_dof_index
  StokesOperator<dim, LinearAlgebra, spacedim>::m() const
  {
    return matrix_free->get_dof_handler(velocity_index).n_dofs() +
           matrix_free->get_dof_handler(pressure_index).n_dofs();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(
    VectorType & /*diagonal*/) const
  {
    Assert(false, ExcNotImplemented());
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  const std::shared_ptr<MatrixFree<dim, typename LinearAlgebra::BlockVector::value_type>> &
  StokesOperator<dim, LinearAlgebra, spacedim>::get_matrix_free() const
  {
    return matrix_free;
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
    FEEvaluation<dim, -1, 0, dim, value_type> velocity(matrix_free, range, velocity_index);
    FEEvaluation<dim, -1, 0, 1, value_type>   pressure(matrix_free, range, pressure_index);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        velocity.reinit(cell);
        velocity.gather_evaluate(src.block(velocity_index), EvaluationFlags::gradients);
        pressure.reinit(cell);
        pressure.gather_evaluate(src.block(pressure_index), EvaluationFlags::values);

        for (unsigned int q = 0; q < velocity.n_q_points; ++q)
          {
            Tensor<1, dim, Tensor<1, dim, VectorizedArray<double>>> grad_u =
              velocity.get_gradient(q);
            VectorizedArray<double> pres  = pressure.get_value(q);
            VectorizedArray<double> div_u = velocity.get_divergence(q);
            pressure.submit_value(-div_u, q);

            // TODO: Move viscosity to class member
            constexpr double viscosity = 0.1;
            grad_u *= viscosity;

            // subtract p * I
            for (unsigned int d = 0; d < dim; ++d)
              grad_u[d][d] -= pres;

            velocity.submit_gradient(grad_u, q);
          }

        velocity.integrate_scatter(EvaluationFlags::gradients, dst.block(velocity_index));
        pressure.integrate_scatter(EvaluationFlags::values, dst.block(pressure_index));
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::do_cell_rhs_function_range(
    const MatrixFree<dim, value_type> &matrix_free,
    VectorType                        &system_rhs,
    const VectorType & /*dummy*/,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FEEvaluation<dim, -1, 0, dim, value_type> velocity(matrix_free, range, velocity_index);
    FEEvaluation<dim, -1, 0, 1, value_type>   pressure(matrix_free, range, pressure_index);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        velocity.reinit(cell);
        pressure.reinit(cell);

        for (unsigned int q = 0; q < velocity.n_q_points; ++q)
          {
            const Point<dim, VectorizedArray<value_type>> p_vect = velocity.quadrature_point(q);
            Tensor<1, dim, VectorizedArray<value_type>>   f_vect;
            for (unsigned int i = 0; i < VectorizedArray<value_type>::size(); ++i)
              {
                Point<dim> p;
                for (unsigned int d = 0; d < dim; ++d)
                  p[d] = p_vect[d][i];
                Vector<value_type> f(dim);
                rhs_functions[velocity_index]->vector_value(p, f);
                for (unsigned int d = 0; d < dim; ++d)
                  f_vect[d][i] = f[d];
              }
            velocity.submit_value(f_vect, q);
          }

        for (unsigned int q = 0; q < pressure.n_q_points; ++q)
          {
            const Point<dim, VectorizedArray<value_type>> p_vect = pressure.quadrature_point(q);
            VectorizedArray<value_type>                   f_vect = 0.;
            for (unsigned int i = 0; i < VectorizedArray<value_type>::size(); ++i)
              {
                Point<dim> p;
                for (unsigned int d = 0; d < dim; ++d)
                  p[d] = p_vect[d][i];
                f_vect[i] = rhs_functions[pressure_index]->value(p);
              }
            pressure.submit_value(f_vect, q);
          }

        velocity.integrate_scatter(EvaluationFlags::values, system_rhs.block(velocity_index));
        pressure.integrate_scatter(EvaluationFlags::values, system_rhs.block(pressure_index));
      }
  }



// explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class ABlockOperator<2, dealiiTrilinos, 2>;
  template class ABlockOperator<3, dealiiTrilinos, 3>;
  template class SchurBlockOperator<2, dealiiTrilinos, 2>;
  template class SchurBlockOperator<3, dealiiTrilinos, 3>;
  template class StokesOperator<2, dealiiTrilinos, 2>;
  template class StokesOperator<3, dealiiTrilinos, 3>;
#endif

} // namespace StokesMatrixFree
