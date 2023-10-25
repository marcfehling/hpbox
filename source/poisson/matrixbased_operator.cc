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
#include <poisson/matrixbased_operator.h>

using namespace dealii;


namespace PoissonMatrixBased
{
  template <int dim, typename LinearAlgebra, int spacedim>
  PoissonOperator<dim, LinearAlgebra, spacedim>::PoissonOperator(
    const hp::MappingCollection<dim, spacedim> &mapping_collection,
    const hp::QCollection<dim>                 &quadrature_collection,
    const hp::FECollection<dim, spacedim>      &fe_collection)
    : mapping_collection(&mapping_collection)
    , quadrature_collection(&quadrature_collection)
    , fe_values_collection(mapping_collection,
                           fe_collection,
                           quadrature_collection,
                           update_values | update_gradients | update_quadrature_points |
                             update_JxW_values)
  {
    TimerOutput::Scope t(getTimer(), "calculate_fevalues");

    fe_values_collection.precalculate_fe_values();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  PoissonOperator<dim, LinearAlgebra, spacedim>::PoissonOperator(
    const hp::MappingCollection<dim, spacedim> &mapping_collection,
    const hp::QCollection<dim>                 &quadrature_collection,
    const hp::FEValues<dim, spacedim>          &fe_values_collection)
    : mapping_collection(&mapping_collection)
    , quadrature_collection(&quadrature_collection)
    , fe_values_collection(fe_values_collection)
  {}



  template <int dim, typename LinearAlgebra, int spacedim>
  std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>
  PoissonOperator<dim, LinearAlgebra, spacedim>::replicate() const
  {
    return std::make_unique<PoissonOperator<dim, LinearAlgebra, spacedim>>(*mapping_collection,
                                                                           *quadrature_collection,
                                                                           fe_values_collection);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::reinit(const Partitioning &,
                                                        const DoFHandler<dim, spacedim> &,
                                                        const AffineConstraints<value_type> &)
  {
    Assert(false, ExcNotImplemented());
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
    {
      TimerOutput::Scope t(getTimer(), "setup_system");

      this->dealii_partitioner =
        std::make_shared<const Utilities::MPI::Partitioner>(partitioning.get_owned_dofs(),
                                                            partitioning.get_relevant_dofs(),
                                                            dof_handler.get_communicator());

      {
        TimerOutput::Scope t(getTimer(), "reinit_matrix");

        initialize_sparse_matrix(system_matrix, dof_handler, constraints, partitioning);
      }

      {
        TimerOutput::Scope t(getTimer(), "reinit_vectors");

        initialize_dof_vector(system_rhs);
      }
    }

    {
      TimerOutput::Scope t(getTimer(), "assemble_system");

      FullMatrix<double>                   cell_matrix;
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> local_dof_indices;
      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned() == false)
            continue;

          const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
          cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
          cell_matrix = 0;
          cell_rhs.reinit(dofs_per_cell);
          cell_rhs = 0;

          fe_values_collection.reinit(cell);
          const FEValues<dim> &fe_values = fe_values_collection.get_present_fe_values();

          for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) += (fe_values.shape_grad(i, q_point) * // grad phi_i(x_q)
                                        fe_values.shape_grad(j, q_point) * // grad phi_j(x_q)
                                        fe_values.JxW(q_point));           // dx
              }
          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(
            cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
        }

      system_rhs.compress(VectorOperation::values::add);
      system_matrix.compress(VectorOperation::values::add);
    }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::vmult(VectorType &dst, const VectorType &src) const
  {
    TimerOutput::Scope t(getTimer(), "vmult");

    system_matrix.vmult(dst, src);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::initialize_dof_vector(VectorType &vec) const
  {
    // LA::distributed::Vector needs to know about ghost indices,
    // but Trilinos/PETSc::MPI::Vector must remain non-ghosted.

    Assert(dealii_partitioner->n_mpi_processes() == 1 || dealii_partitioner->ghost_indices_initialized(), ExcInternalError());
    vec.reinit(dealii_partitioner, /*make_ghosted*/ false);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  types::global_dof_index
  PoissonOperator<dim, LinearAlgebra, spacedim>::m() const
  {
    return system_matrix.m();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(
    VectorType &diagonal) const
  {
    this->initialize_dof_vector(diagonal);

    for (const auto n : diagonal.locally_owned_elements())
      diagonal[n] = 1.0 / system_matrix.diag_element(n);

    diagonal.compress(VectorOperation::values::insert);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  const typename LinearAlgebra::SparseMatrix &
  PoissonOperator<dim, LinearAlgebra, spacedim>::get_system_matrix() const
  {
    return system_matrix;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  PoissonOperator<dim, LinearAlgebra, spacedim>::Tvmult(VectorType       &dst,
                                                        const VectorType &src) const
  {
    vmult(dst, src);
  }



// explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class PoissonOperator<2, dealiiTrilinos, 2>;
  template class PoissonOperator<3, dealiiTrilinos, 3>;
  template class PoissonOperator<2, Trilinos, 2>;
  template class PoissonOperator<3, Trilinos, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
  template class PoissonOperator<2, PETSc, 2>;
  template class PoissonOperator<3, PETSc, 3>;
#endif

} // namespace PoissonMatrixBased
