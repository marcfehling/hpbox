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
#include <stokes/matrixbased/stokes_operator.h>

using namespace dealii;


namespace StokesMatrixBased
{
  template <int dim, typename LinearAlgebra, int spacedim>
  StokesOperator<dim, LinearAlgebra, spacedim>::StokesOperator(
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
  StokesOperator<dim, LinearAlgebra, spacedim>::StokesOperator(
    const hp::MappingCollection<dim, spacedim> &mapping_collection,
    const hp::QCollection<dim>                 &quadrature_collection,
    const hp::FEValues<dim, spacedim>          &fe_values_collection)
    : mapping_collection(&mapping_collection)
    , quadrature_collection(&quadrature_collection)
    , fe_values_collection(fe_values_collection)
  {}


  template <int dim, typename LinearAlgebra, int spacedim>
  std::unique_ptr<BlockOperatorType<dim, LinearAlgebra, spacedim>>
  StokesOperator<dim, LinearAlgebra, spacedim>::replicate() const
  {
    return std::make_unique<StokesOperator<dim, LinearAlgebra, spacedim>>(*mapping_collection,
                                                                          *quadrature_collection,
                                                                          fe_values_collection);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::reinit(const Partitioning &,
                                                       const DoFHandler<dim, spacedim> &,
                                                       const AffineConstraints<value_type> &)
  {
    Assert(false, ExcNotImplemented());
  }


  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::reinit(
    const Partitioning                  &partitioning,
    const DoFHandler<dim, spacedim>     &dof_handler,
    const AffineConstraints<value_type> &constraints,
    VectorType                          &system_rhs,
    const dealii::Function<spacedim>    *rhs_function)
  {
    {
      TimerOutput::Scope t(getTimer(), "setup_system");

      // setup partitioners for initialize_dof_vector
      this->communicator = dof_handler.get_communicator();
      this->partitioning = partitioning;

      if constexpr (std::is_same_v<typename LinearAlgebra::BlockVector,
                                   dealii::LinearAlgebra::distributed::BlockVector<double>>)
        {
          dealii_partitioners.clear();
          dealii_partitioners.reserve(partitioning.get_n_blocks());
          for (unsigned int b = 0; b < partitioning.get_n_blocks(); ++b)
            {
              const IndexSet &owned_dofs    = partitioning.get_owned_dofs_per_block()[b];
              const IndexSet &relevant_dofs = partitioning.get_relevant_dofs_per_block()[b];

              dealii_partitioners.emplace_back(std::make_shared<Utilities::MPI::Partitioner>(
                owned_dofs, relevant_dofs, communicator));
            }
        }

      {
        TimerOutput::Scope t(getTimer(), "reinit_matrices");

        system_matrix.clear();

        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
        for (unsigned int c = 0; c < dim + 1; ++c)
          for (unsigned int d = 0; d < dim + 1; ++d)
            if (!((c == dim) && (d == dim)))
              coupling[c][d] = DoFTools::always;
            else
              coupling[c][d] = DoFTools::none;

        initialize_block_sparse_matrix(
          system_matrix, dof_handler, constraints, partitioning, coupling);
      }

      {
        TimerOutput::Scope t(getTimer(), "reinit_vectors");

        initialize_dof_vector(system_rhs);
      }

      {
        TimerOutput::Scope t(getTimer(), "assemble_system");

        // system_matrix         = 0;
        // system_rhs            = 0;

        FullMatrix<double> cell_matrix;
        Vector<double>     cell_rhs;

        std::vector<Vector<double>> rhs_values;

        std::vector<Tensor<2, dim>> grad_phi_u;
        std::vector<double>         div_phi_u;
        std::vector<double>         phi_p;

        std::vector<types::global_dof_index> local_dof_indices;
        const FEValuesExtractors::Vector     velocities(0);
        const FEValuesExtractors::Scalar     pressure(dim);
        for (const auto &cell :
             dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
          {
            fe_values_collection.reinit(cell);

            const FEValues<dim> &fe_values     = fe_values_collection.get_present_fe_values();
            const unsigned int   n_q_points    = fe_values.n_quadrature_points;
            const unsigned int   dofs_per_cell = fe_values.dofs_per_cell;

            cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
            cell_matrix = 0;
            cell_rhs.reinit(dofs_per_cell);
            cell_rhs = 0;

            grad_phi_u.resize(dofs_per_cell);
            div_phi_u.resize(dofs_per_cell);
            phi_p.resize(dofs_per_cell);

            local_dof_indices.resize(dofs_per_cell);

            // TODO: Move this part to the problem class???
            //       Not possible...
            rhs_values.resize(n_q_points, Vector<double>(dim + 1));

            // TODO: Make rhs function a parameter?
            if (rhs_function != nullptr)
              rhs_function->vector_value_list(fe_values.get_quadrature_points(), rhs_values);

            // TODO: move to parameter
            const double viscosity = 0.1;

            for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
              {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    grad_phi_u[k] = fe_values[velocities].gradient(k, q_point);
                    div_phi_u[k]  = fe_values[velocities].divergence(k, q_point);
                    phi_p[k]      = fe_values[pressure].value(k, q_point);
                  }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        cell_matrix(i, j) +=
                          (viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]) -
                           div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
                          fe_values.JxW(q_point);
                      }

                    const unsigned int component_i =
                      cell->get_fe().system_to_component_index(i).first;
                    cell_rhs(i) += fe_values.shape_value(i, q_point) *
                                   rhs_values[q_point](component_i) * fe_values.JxW(q_point);
                  }
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
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::vmult(VectorType &dst, const VectorType &src) const
  {
    TimerOutput::Scope t(getTimer(), "vmult");

    system_matrix.vmult(dst, src);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::initialize_dof_vector(VectorType &vec) const
  {
    if constexpr (std::is_same_v<typename LinearAlgebra::BlockVector,
                                 dealii::LinearAlgebra::distributed::BlockVector<double>>)
      {
        // LA::distributed::BlockVector must remain non-ghosted, but needs to know about ghost
        // indices

#ifdef DEBUG
        for (const auto &partitioner : dealii_partitioners)
          Assert(partitioner->ghost_indices_initialized(), ExcInternalError());
#endif

        // TODO: available with #14595
        // vec.reinit(dealii_partitioners);

        vec.reinit(dealii_partitioners.size());
        for (unsigned int b = 0; b < vec.n_blocks(); ++b)
          vec.block(b).reinit(dealii_partitioners[b]);
        vec.collect_sizes();
      }
    else
      {
        // Trilinos/PETSc::MPI::BlockVector must remain non-ghosted
        vec.reinit(partitioning.get_owned_dofs_per_block(), communicator);
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  types::global_dof_index
  StokesOperator<dim, LinearAlgebra, spacedim>::m() const
  {
    return system_matrix.m();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(VectorType &diagonal) const
  {
    this->initialize_dof_vector(diagonal);

    for (unsigned int n = 0; n < system_matrix.n(); ++n)
      diagonal[n] = 1.0 / system_matrix.diag_element(n);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  const typename LinearAlgebra::BlockSparseMatrix &
  StokesOperator<dim, LinearAlgebra, spacedim>::get_system_matrix() const
  {
    return system_matrix;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  StokesOperator<dim, LinearAlgebra, spacedim>::Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }



// explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class StokesOperator<2, dealiiTrilinos, 2>;
  template class StokesOperator<3, dealiiTrilinos, 3>;
  template class StokesOperator<2, Trilinos, 2>;
  template class StokesOperator<3, Trilinos, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
  template class StokesOperator<2, PETSc, 2>;
  template class StokesOperator<3, PETSc, 3>;
#endif

} // namespace StokesMatrixBased
