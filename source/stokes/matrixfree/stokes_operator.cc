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
    const std::vector<const DoFHandler<dim, spacedim> *>     &dof_handlers,
    const std::vector<const AffineConstraints<value_type> *> &constraints,
    VectorType                                               &system_rhs,
    const std::vector<const dealii::Function<spacedim> *>    &rhs_functions)
  {
    TimerOutput::Scope t(getTimer(), "reinit");

    this->constraints = &constraints;

    typename MatrixFree<dim, value_type>::AdditionalData data;
    data.mapping_update_flags = update_gradients;
    // TODO: more?

    matrix_free.reinit(*mapping_collection, dof_handlers, constraints, *quadrature_collections, data);


    this->initialize_dof_vector(system_rhs);

    // !!! TODO !!!
    // add rhs here
    // how to do this in a matrix free fashion? --- check tutorials (step-67)

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
    FEEvaluation<dim, -1, 0, dim, value_type> velocity (matrix_free, 0);
    FEEvaluation<dim, -1, 0, 1  , value_type> pressure (matrix_free, 1);

    for (unsigned int cell=range.first; cell<range.second; ++cell)
      {
        velocity.reinit (cell);
        velocity.read_dof_values (src.block(0));
        velocity.evaluate (EvaluationFlags::gradients);
        pressure.reinit (cell);
        pressure.read_dof_values (src.block(1));
        pressure.evaluate (EvaluationFlags::values);

        for (unsigned int q=0; q<velocity.n_q_points; ++q)
          {
            SymmetricTensor<2,dim,VectorizedArray<double> > sym_grad_u =
              velocity.get_symmetric_gradient (q);
            VectorizedArray<double> pres = pressure.get_value(q);
            VectorizedArray<double> div = -trace(sym_grad_u);
            pressure.submit_value (div, q);

            // subtract p * I
            for (unsigned int d=0; d<dim; ++d)
              sym_grad_u[d][d] -= pres;

            velocity.submit_symmetric_gradient(sym_grad_u, q);
         }

        velocity.integrate (EvaluationFlags::gradients);
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
