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


#include <deal.II/distributed/tria_base.h>


template <typename MeshType>
MPI_Comm
get_mpi_comm(const MeshType &mesh)
{
  const auto *tria_parallel = dynamic_cast<
    const dealii::parallel::TriangulationBase<MeshType::dimension,
                                              MeshType::space_dimension> *>(
    &(mesh.get_triangulation()));

  return tria_parallel != nullptr ? tria_parallel->get_communicator() :
                                    MPI_COMM_SELF;
}


// TODO: No need for this base class anymore since dealii::MGSolverOperatorBase
//       has been introduced.
/*
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/multigrid/mg_solver.h>

namespace Operator
{
  template <int dim, typename VectorType>
  class Base : public dealii::MGSolverOperatorBase<dim, typename VectorType::value_type>
  {
  public:
    using value_type = typename VectorType::value_type;

    virtual void
    reinit(const dealii::hp::MappingCollection<dim> &   mapping_collection,
           const dealii::DoFHandler<dim> &              dof_handler,
           const dealii::hp::QCollection<dim> &         quadrature_collection,
           const dealii::AffineConstraints<value_type> &constraints,
           VectorType &                                 system_rhs) = 0;
  };
}
*/


#endif
