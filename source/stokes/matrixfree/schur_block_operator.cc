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
    const hp::QCollection<dim>                 &quadrature_collection)
    : mapping_collection(&mapping_collection)
    , quadrature_collection(&quadrature_collection)
  {}


  template <int dim, typename LinearAlgebra, int spacedim>
  std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::replicate() const
  {
    return std::make_unique<SchurBlockOperator<dim, LinearAlgebra, spacedim>>(
      *mapping_collection, *quadrature_collection);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::reinit(
    const Partitioning                  &partitioning,
    const DoFHandler<dim, spacedim>     &dof_handler,
    const AffineConstraints<value_type> &constraints)
  {

    TimerOutput::Scope t(getTimer(), "setup_system");

    this->schur_block_matrix.clear();

    this->partitioning = partitioning;
    this->constraints  = &constraints;

    typename MatrixFree<dim, value_type>::AdditionalData data;
    data.mapping_update_flags = update_gradients;

    matrix_free.reinit(*mapping_collection, dof_handler, constraints, *quadrature_collection, data);
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
    this->matrix_free.cell_loop(&SchurBlockOperator::do_cell_integral_range, this, dst, src, true);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  types::global_dof_index
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::m() const
  {
    return matrix_free.get_dof_handler().n_dofs();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  SchurBlockOperator<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(
    VectorType &diagonal) const
  {
    matrix_free.initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(matrix_free,
                                      diagonal,
                                      &SchurBlockOperator::do_cell_integral_local,
                                      this);

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
        const auto &dof_handler = this->matrix_free.get_dof_handler();

        initialize_sparse_matrix(schur_block_matrix, dof_handler, *constraints, partitioning);

        MatrixFreeTools::compute_matrix(
          matrix_free, *constraints, schur_block_matrix, &SchurBlockOperator::do_cell_integral_local, this);
      }

    return this->schur_block_matrix;
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
        constexpr double viscosity = 0.1;
        constexpr double inv_viscosity = 1/viscosity;
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
        constexpr double viscosity = 0.1;
        constexpr double inv_viscosity = 1/viscosity;
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
    FECellIntegrator pressure(matrix_free, range);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        pressure.reinit(cell);

        do_cell_integral_global(pressure, dst, src);
      }
  }



// explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class SchurBlockOperator<2, dealiiTrilinos, 2>;
  template class SchurBlockOperator<3, dealiiTrilinos, 3>;
#endif

} // namespace StokesMatrixFree
