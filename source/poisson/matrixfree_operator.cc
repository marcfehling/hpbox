// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2023 by the deal.II authors
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


#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <global.h>
#include <linear_algebra.h>
#include <poisson/matrixfree_operator.h>

using namespace dealii;


namespace PoissonMatrixFree
{
  template <int dim, typename LinearAlgebra, int spacedim>
  PoissonOperator<dim, LinearAlgebra, spacedim>::PoissonOperator(
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
  PoissonOperator<dim, LinearAlgebra, spacedim>::replicate() const
  {
    return std::make_unique<PoissonOperator<dim, LinearAlgebra, spacedim>>(
      *mapping_collection, *quadrature_collection, hp::FECollection<dim, spacedim>());
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::reinit(
    const Partitioning                  &partitioning,
    const DoFHandler<dim, spacedim>     &dof_handler,
    const AffineConstraints<value_type> &constraints)
  {
    this->system_matrix.clear();

    this->partitioning = partitioning;
    this->constraints  = &constraints;

    typename MatrixFree<dim, value_type>::AdditionalData data;
    data.mapping_update_flags = update_gradients;

    matrix_free.reinit(*mapping_collection, dof_handler, constraints, *quadrature_collection, data);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::reinit(
    const Partitioning                  &partitioning,
    const DoFHandler<dim, spacedim>     &dof_handler,
    const AffineConstraints<value_type> &constraints,
    VectorType                          &system_rhs,
    const dealii::Function<spacedim> * /*rhs_function*/)
  {
    TimerOutput::Scope t(getTimer(), "reinit");

    this->system_matrix.clear();

    this->partitioning = partitioning;
    this->constraints  = &constraints;

    typename MatrixFree<dim, value_type>::AdditionalData data;
    data.mapping_update_flags = update_gradients;

    matrix_free.reinit(*mapping_collection, dof_handler, constraints, *quadrature_collection, data);


    this->initialize_dof_vector(system_rhs);
    {
      AffineConstraints<value_type> constraints_without_dbc;
      constraints_without_dbc.reinit(partitioning.get_relevant_dofs());
      DoFTools::make_hanging_node_constraints(dof_handler, constraints_without_dbc);
      constraints_without_dbc.make_consistent_in_parallel(partitioning.get_owned_dofs(),
                                                          partitioning.get_active_dofs(),
                                                          partitioning.get_communicator());
      constraints_without_dbc.close();

      VectorType b, x;

      MatrixFree<dim, value_type> matrix_free;
      matrix_free.reinit(
        *mapping_collection, dof_handler, constraints_without_dbc, *quadrature_collection, data);

      matrix_free.initialize_dof_vector(b);
      matrix_free.initialize_dof_vector(x);

      constraints.distribute(x);

      matrix_free.cell_loop(&PoissonOperator::do_cell_integral_range, this, b, x);

      constraints.set_zero(b);

      system_rhs -= b;
    }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::vmult(VectorType &dst, const VectorType &src) const
  {
    TimerOutput::Scope t(getTimer(), "vmult");

    this->matrix_free.cell_loop(&PoissonOperator::do_cell_integral_range, this, dst, src, true);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  types::global_dof_index
  PoissonOperator<dim, LinearAlgebra, spacedim>::m() const
  {
    return matrix_free.get_dof_handler().n_dofs();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(
    VectorType &diagonal) const
  {
    matrix_free.initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(matrix_free,
                                      diagonal,
                                      &PoissonOperator::do_cell_integral_local,
                                      this);

    // invert diagonal
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::compute_lumped_inverse_diagonal(
    VectorType &) const
  {
    Assert(false, ExcNotImplemented());
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  const typename LinearAlgebra::SparseMatrix &
  PoissonOperator<dim, LinearAlgebra, spacedim>::get_system_matrix() const
  {
    // Check if matrix has already been set up.
    if (system_matrix.m() == 0 && system_matrix.n() == 0)
      {
        const auto &dof_handler = this->matrix_free.get_dof_handler();

        initialize_sparse_matrix(system_matrix, dof_handler, *constraints, partitioning);

        MatrixFreeTools::compute_matrix(
          matrix_free, *constraints, system_matrix, &PoissonOperator::do_cell_integral_local, this);
      }

    return this->system_matrix;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::Tvmult(VectorType       &dst,
                                                        const VectorType &src) const
  {
    this->vmult(dst, src);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::do_cell_integral_local(
    FECellIntegrator &integrator) const
  {
    integrator.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate(EvaluationFlags::gradients);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::do_cell_integral_global(
    FECellIntegrator &integrator,
    VectorType       &dst,
    const VectorType &src) const
  {
    integrator.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate_scatter(EvaluationFlags::gradients, dst);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::do_cell_integral_range(
    const MatrixFree<dim, value_type>           &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free, range);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);

        do_cell_integral_global(integrator, dst, src);
      }
  }



  // explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class PoissonOperator<2, dealiiTrilinos, 2>;
  template class PoissonOperator<3, dealiiTrilinos, 3>;
#endif

} // namespace PoissonMatrixFree
