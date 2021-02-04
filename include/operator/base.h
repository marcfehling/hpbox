// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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

#ifndef operator_base_h
#define operator_base_h


#include <deal.II/base/subscriptor.h>
#include <deal.II/base/types.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>


namespace Operator
{
  template <int dim, typename VectorType>
  class Base : public dealii::Subscriptor
  {
  public:
    using value_type = typename VectorType::value_type;

    virtual void
    reinit(const dealii::hp::MappingCollection<dim> &mapping_collection,
           const dealii::DoFHandler<dim> &           dof_handler,
           const dealii::hp::QCollection<dim> &      quadrature_collection,
           const dealii::AffineConstraints<value_type> & constraints,
           VectorType &                      system_rhs) = 0;

    virtual void
    vmult(VectorType &dst, const VectorType &src) const = 0;

    virtual void
    initialize_dof_vector(VectorType &vec) const = 0;

    virtual dealii::types::global_dof_index m() const = 0;

    virtual void compute_inverse_diagonal(VectorType &diagonal) const = 0;

    virtual const dealii::TrilinosWrappers::SparseMatrix &get_system_matrix() const = 0;

    virtual void Tvmult(VectorType &dst, const VectorType &src) const = 0;
  };
}


#endif
