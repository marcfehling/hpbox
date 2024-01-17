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

#ifndef poisson_matrixbased_operator_h
#define poisson_matrixbased_operator_h


#include <deal.II/base/partitioner.h>

#include <deal.II/hp/fe_values.h>

#include <multigrid/operator_base.h>


namespace PoissonMatrixBased
{
  template <int dim, typename LinearAlgebra, int spacedim = dim>
  class PoissonOperator : public OperatorType<dim, LinearAlgebra, spacedim>
  {
  public:
    using VectorType = typename LinearAlgebra::Vector;
    using value_type = typename VectorType::value_type;

    PoissonOperator(const dealii::hp::MappingCollection<dim, spacedim> &mapping_collection,
                    const dealii::hp::QCollection<dim>                 &quadrature_collection,
                    const dealii::hp::FECollection<dim, spacedim>      &fe_collection);

    PoissonOperator(const dealii::hp::MappingCollection<dim, spacedim> &mapping_collection,
                    const dealii::hp::QCollection<dim>                 &quadrature_collection,
                    const dealii::hp::FEValues<dim, spacedim>          &fe_values_collection);

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

    // TODO: Add RHS function to constructor
    //       Grab and set as RHS in reinit
    // dealii::Function<dim> rhs_function;

    dealii::hp::FEValues<dim, spacedim> fe_values_collection;

    typename LinearAlgebra::SparseMatrix system_matrix;

    std::shared_ptr<const dealii::Utilities::MPI::Partitioner> dealii_partitioner;
  };
} // namespace PoissonMatrixBased


#endif
