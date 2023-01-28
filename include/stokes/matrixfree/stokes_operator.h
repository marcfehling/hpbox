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

#ifndef stokes_matrixfree_stokes_operator_h
#define stokes_matrixfree_stokes_operator_h


#include <deal.II/matrix_free/tools.h>

#include <operator.h>


namespace StokesMatrixFree
{
  template <int dim, typename LinearAlgebra, int spacedim = dim>
  class StokesOperator
    : public dealii::MGSolverOperatorBase<dim,
                                          typename LinearAlgebra::BlockVector,
                                          typename LinearAlgebra::BlockSparseMatrix>
  {
  public:
    using VectorType = typename LinearAlgebra::BlockVector;
    using value_type = typename VectorType::value_type;

    // using FECellIntegrator = dealii::FEEvaluation<dim, -1, 0, dim + 1, value_type>;

    StokesOperator(const dealii::hp::MappingCollection<dim, spacedim> &mapping_collection,
                   const std::vector<dealii::hp::QCollection<dim>>    &quadrature_collections);

    void
    reinit(const std::vector<const dealii::DoFHandler<dim, spacedim> *>     &dof_handlers,
           const std::vector<const dealii::AffineConstraints<value_type> *> &constraints,
           VectorType                                                       &system_rhs,
           const std::vector<const dealii::Function<spacedim> *>            &rhs_functions);

    void
    vmult(VectorType &dst, const VectorType &src) const override;

    void
    initialize_dof_vector(VectorType &vec) const override;

    dealii::types::global_dof_index
    m() const override;

    void
    compute_inverse_diagonal(VectorType &diagonal) const override;

    const typename LinearAlgebra::BlockSparseMatrix &
    get_system_matrix() const override;

    void
    Tvmult(VectorType &dst, const VectorType &src) const override;

  private:
    // const Parameters &prm;

    dealii::SmartPointer<const dealii::hp::MappingCollection<dim, spacedim>> mapping_collection;
    const std::vector<dealii::hp::QCollection<dim>> * quadrature_collections;
    const std::vector<const dealii::AffineConstraints<value_type> *> * constraints;

    // TODO: Add RHS function to constructor
    //       Grab and set as RHS in reinit
    // dealii::Function<dim> rhs_function;

    void
    do_cell_integral_range(const dealii::MatrixFree<dim, value_type>   &matrix_free,
                           VectorType                                  &dst,
                           const VectorType                            &src,
                           const std::pair<unsigned int, unsigned int> &range) const;

    // TODO: Make partitioning a pointer? Or leave it like this?
    Partitioning                        partitioning;
    dealii::MatrixFree<dim, value_type> matrix_free;

    mutable typename LinearAlgebra::BlockSparseMatrix dummy;
  };
} // namespace StokesMatrixFree


#endif
