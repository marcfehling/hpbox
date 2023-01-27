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
    const Partitioning                                 &partitioning,
    const std::vector<DoFHandler<dim, spacedim> *>     &dof_handlers,
    const std::vector<AffineConstraints<value_type> *> &constraints,
    VectorType                                         &system_rhs,
    const dealii::Function<spacedim>                   *rhs_function)
  {
    TimerOutput::Scope t(getTimer(), "reinit");

    this->partitioning = partitioning;
    this->constraints =
      std::shared_ptr<const std::vector<AffineConstraints<value_type> *>>(&constraints);

    typename MatrixFree<dim, value_type>::AdditionalData data;
    data.mapping_update_flags = update_gradients;
    // TODO: more?

    // matrix_free.reinit(mapping_collection.get(), dof_handlers, constraints, quadrature_collections.get(), data);


    this->initialize_dof_vector(system_rhs);

    // !!! TODO !!!
    // add rhs here
    // how to do this in a matrix free fashion? --- check tutorials (step-67)
    (void)rhs_function;

    {
      // AffineConstraints<value_type> constraints_without_dbc;

      // constraints_without_dbc.reinit(partitioning.get_relevant_dofs());

      // DoFTools::make_hanging_node_constraints(dof_handler, constraints_without_dbc);
      // constraints_without_dbc.close();

      // VectorType b, x;

      // MatrixFree<dim, value_type> matrix_free;
      // matrix_free.reinit(
      //   *mapping_collection, dof_handler, constraints_without_dbc, *quadrature_collection, data);

      // // matrix_free.initialize_dof_vector(b);
      // // matrix_free.initialize_dof_vector(x);
      // this->initialize_dof_vector(b);
      // this->initialize_dof_vector(x);

      // constraints.distribute(x);

      // matrix_free.cell_loop(&StokesOperator::do_cell_integral_range, this, b, x);

      // constraints.set_zero(b);

      // system_rhs -= b;
    }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::vmult(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(&StokesOperator::do_cell_integral_range, this, dst, src, true);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec.block(0), 0);
    matrix_free.initialize_dof_vector(vec.block(1), 1);
    vec.collect_sizes();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  types::global_dof_index
  StokesOperator<dim, LinearAlgebra, spacedim>::m() const
  {
    // TODO
    // of each dofhandler
    return matrix_free.get_dof_handler().n_dofs();
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
    Assert(false, ExcNotImplemented());

    (void)dst;
    (void)src;

    // FECellIntegrator integrator(matrix_free, range);

    for (unsigned int cell = range.first; cell < range.second; ++cell)
      {
        (void)cell;
        // integrator.reinit(cell);

        // ...
      }
  }



// explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class StokesOperator<2, dealiiTrilinos, 2>;
  template class StokesOperator<3, dealiiTrilinos, 3>;
#endif

} // namespace StokesMatrixFree
