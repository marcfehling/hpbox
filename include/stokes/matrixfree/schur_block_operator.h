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

#ifndef stokes_matrixfree_schur_block_operator_h
#define stokes_matrixfree_schur_block_operator_h


#include <deal.II/lac/diagonal_matrix.h>

#include <deal.II/matrix_free/tools.h>

#include <operator.h>


namespace StokesMatrixFree
{
  template <int dim, typename LinearAlgebra, int spacedim = dim>
  class SchurBlockOperator : public OperatorType<dim, LinearAlgebra, spacedim>
  {
  public:
    using VectorType = typename LinearAlgebra::Vector;
    using value_type = typename VectorType::value_type;

    using FECellIntegrator = dealii::FEEvaluation<dim, -1, 0, 1, value_type>;

    SchurBlockOperator(const dealii::hp::MappingCollection<dim, spacedim> &mapping_collection,
                       const dealii::hp::QCollection<dim>                 &quadrature_collection);

    std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>
    replicate() const override;

    void
    reinit(const Partitioning                          &partitioning,
           const dealii::DoFHandler<dim, spacedim>     &dof_handler,
           const dealii::AffineConstraints<value_type> &constraints) override;

    void
    reinit(const Partitioning                          &partitioning,
           const dealii::DoFHandler<dim, spacedim>     &dof_handler,
           const dealii::AffineConstraints<value_type> &constraints,
           VectorType                                  &system_rhs,
           const dealii::Function<spacedim>            *rhs_function) override;

    void
    vmult(VectorType &dst, const VectorType &src) const override;

    void
    initialize_dof_vector(VectorType &vec) const override;

    dealii::types::global_dof_index
    m() const override;

    void
    compute_inverse_diagonal(VectorType &diagonal) const override;

    void
    precondition_Jacobi(VectorType       &dst,
                        const VectorType &src,
                        const value_type omega) const;

    const typename LinearAlgebra::SparseMatrix &
    get_system_matrix() const override;

    void
    Tvmult(VectorType &dst, const VectorType &src) const override;

  private:
    // const Parameters &prm;

    dealii::SmartPointer<const dealii::hp::MappingCollection<dim, spacedim>> mapping_collection;
    dealii::SmartPointer<const dealii::hp::QCollection<dim>>                 quadrature_collection;
    dealii::SmartPointer<const dealii::AffineConstraints<value_type>>        constraints;

    void
    do_cell_integral_local(FECellIntegrator &integrator) const;

    void
    do_cell_integral_global(FECellIntegrator &integrator,
                            VectorType       &dst,
                            const VectorType &src) const;

    void
    do_cell_integral_range(const dealii::MatrixFree<dim, value_type>   &matrix_free,
                           VectorType                                  &dst,
                           const VectorType                            &src,
                           const std::pair<unsigned int, unsigned int> &range) const;

    // TODO: Add RHS function to constructor
    //       Grab and set as RHS in reinit
    // dealii::Function<dim> rhs_function;

    Partitioning                        partitioning;
    dealii::MatrixFree<dim, value_type> matrix_free;

    mutable typename LinearAlgebra::SparseMatrix schur_block_matrix;

    // test jacobi
    mutable dealii::DiagonalMatrix<VectorType> diagonal_matrix;
  };
} // namespace StokesMatrixFree


#endif
