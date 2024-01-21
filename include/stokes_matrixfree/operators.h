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

#ifndef stokes_matrixfree_operators_h
#define stokes_matrixfree_operators_h


#include <deal.II/matrix_free/tools.h>

#include <operator_base.h>


namespace StokesMatrixFree
{

  template <int dim, typename LinearAlgebra, int spacedim = dim>
  class ABlockOperator : public OperatorType<dim, LinearAlgebra, spacedim>
  {
  public:
    using VectorType = typename LinearAlgebra::Vector;
    using value_type = typename VectorType::value_type;

    using FECellIntegrator = dealii::FEEvaluation<dim, -1, 0, dim, value_type>;

    ABlockOperator(const dealii::hp::MappingCollection<dim, spacedim> &mapping_collection,
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

    const typename LinearAlgebra::SparseMatrix &
    get_system_matrix() const override;

    void
    Tvmult(VectorType &dst, const VectorType &src) const override;

  private:
    // const Parameters &prm;

    dealii::SmartPointer<const dealii::hp::MappingCollection<dim, spacedim>> mapping_collection;
    dealii::SmartPointer<const dealii::hp::QCollection<dim>>                 quadrature_collection;
    dealii::SmartPointer<const dealii::AffineConstraints<value_type>>        constraints;

    // TODO: Add RHS function to constructor
    //       Grab and set as RHS in reinit
    // dealii::Function<dim> rhs_function;

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

    // TODO: Make partitioning a pointer? Or leave it like this?
    Partitioning                        partitioning;
    dealii::MatrixFree<dim, value_type> matrix_free;

    mutable typename LinearAlgebra::SparseMatrix a_block_matrix;
  };


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
  };



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
    reinit(const std::vector<const Partitioning *>                          &partitionings,
           const std::vector<const dealii::DoFHandler<dim, spacedim> *>     &dof_handlers,
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
    const std::vector<dealii::hp::QCollection<dim>>                         *quadrature_collections;
    const std::vector<const dealii::AffineConstraints<value_type> *>        *constraints;
    const std::vector<const dealii::Function<spacedim> *>                   *rhs_functions;

    void
    do_cell_integral_range(const dealii::MatrixFree<dim, value_type>   &matrix_free,
                           VectorType                                  &dst,
                           const VectorType                            &src,
                           const std::pair<unsigned int, unsigned int> &range) const;

    void
    do_cell_rhs_function_range(const dealii::MatrixFree<dim, value_type> &matrix_free,
                               VectorType                                &system_rhs,
                               const VectorType & /*dummy*/,
                               const std::pair<unsigned int, unsigned int> &range) const;

    // TODO: Make partitioning a pointer? Or leave it like this?
    Partitioning                        partitioning;
    dealii::MatrixFree<dim, value_type> matrix_free;

    mutable typename LinearAlgebra::BlockSparseMatrix dummy;
  };
} // namespace StokesMatrixFree


#endif
