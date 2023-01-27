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
#include <stokes/matrixfree/a_block_operator.h>

using namespace dealii;


namespace StokesMatrixFree
{
  template <int dim, typename LinearAlgebra, int spacedim>
  ABlockOperator<dim, LinearAlgebra, spacedim>::ABlockOperator(
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
  ABlockOperator<dim, LinearAlgebra, spacedim>::ABlockOperator(
    const hp::MappingCollection<dim, spacedim> &mapping_collection,
    const hp::QCollection<dim>                 &quadrature_collection,
    const hp::FEValues<dim, spacedim>          &fe_values_collection)
    : mapping_collection(&mapping_collection)
    , quadrature_collection(&quadrature_collection)
    , fe_values_collection(fe_values_collection)
  {}


  template <int dim, typename LinearAlgebra, int spacedim>
  std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>
  ABlockOperator<dim, LinearAlgebra, spacedim>::replicate() const
  {
    return std::make_unique<ABlockOperator<dim, LinearAlgebra, spacedim>>(*mapping_collection,
                                                                          *quadrature_collection,
                                                                          fe_values_collection);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::reinit(
    const Partitioning                  &partitioning,
    const DoFHandler<dim, spacedim>     &dof_handler,
    const AffineConstraints<value_type> &constraints)
  {
    {
      TimerOutput::Scope t(getTimer(), "setup_system");

      // setup partitioners for initialize_dof_vector
      this->communicator = dof_handler.get_communicator();
      this->partitioning = partitioning;

      // TODO: this is different compared to matrixbased
      this->dealii_partitioner = std::make_shared<const Utilities::MPI::Partitioner>(
        partitioning.get_owned_dofs(),
        partitioning.get_relevant_dofs(),
        dof_handler.get_communicator());

      {
        TimerOutput::Scope t(getTimer(), "reinit_matrices");

        initialize_sparse_matrix(
          a_block_matrix, dof_handler, constraints, partitioning);
      }

      {
        TimerOutput::Scope t(getTimer(), "assemble_system");

        // system_matrix         = 0;
        // preconditioner_matrix = 0;
        // system_rhs            = 0;

        FullMatrix<double> cell_matrix;

        std::vector<Tensor<2, dim>> grad_phi_u;

        std::vector<types::global_dof_index> local_dof_indices;
        const FEValuesExtractors::Vector     velocities(0);
        for (const auto &cell :
             dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
          {
            fe_values_collection.reinit(cell);

            const FEValues<dim> &fe_values     = fe_values_collection.get_present_fe_values();
            const unsigned int   n_q_points    = fe_values.n_quadrature_points;
            const unsigned int   dofs_per_cell = fe_values.dofs_per_cell;

            cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
            cell_matrix = 0;

            grad_phi_u.resize(dofs_per_cell);

            // TODO: move to parameter
            const double viscosity = 0.1;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  grad_phi_u[k] = fe_values[velocities].gradient(k, q_point);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]) *
                                         fe_values.JxW(q_point);
              }

            local_dof_indices.resize(dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);

            constraints.distribute_local_to_global(cell_matrix, local_dof_indices, a_block_matrix);
          }

        a_block_matrix.compress(VectorOperation::values::add);
      }
    }
  }


  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::reinit(const Partitioning &,
                                                       const DoFHandler<dim, spacedim> &,
                                                       const AffineConstraints<value_type> &,
                                                       VectorType &,
                                                       const dealii::Function<spacedim> *)
  {
    Assert(false, ExcNotImplemented());
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::vmult(VectorType &dst, const VectorType &src) const
  {
    a_block_matrix.vmult(dst, src);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::initialize_dof_vector(VectorType &vec) const
  {
    if constexpr (std::is_same_v<typename LinearAlgebra::Vector,
                                 dealii::LinearAlgebra::distributed::Vector<double>>)
      {
        // LA::distributed::Vector must remain non-ghosted, but needs to know about ghost indices
        Assert(dealii_partitioner->ghost_indices_initialized(), ExcInternalError());
        vec.reinit(dealii_partitioner);
      }
    else
      {
        // Trilinos/PETSc::MPI::Vector must remain non-ghosted
        vec.reinit(dealii_partitioner->locally_owned_range(),
                   dealii_partitioner->get_mpi_communicator());
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  types::global_dof_index
  ABlockOperator<dim, LinearAlgebra, spacedim>::m() const
  {
    return a_block_matrix.m();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(VectorType &diagonal) const
  {
    this->initialize_dof_vector(diagonal);

    for (unsigned int n = 0; n < a_block_matrix.n(); ++n)
      diagonal[n] = 1.0 / a_block_matrix.diag_element(n);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  const typename LinearAlgebra::SparseMatrix &
  ABlockOperator<dim, LinearAlgebra, spacedim>::get_system_matrix() const
  {
    return a_block_matrix;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  ABlockOperator<dim, LinearAlgebra, spacedim>::Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }



// explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class ABlockOperator<2, dealiiTrilinos, 2>;
  template class ABlockOperator<3, dealiiTrilinos, 3>;
#endif

} // namespace StokesMatrixFree
