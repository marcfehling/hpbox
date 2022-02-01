// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
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


#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/filtered_iterator.h>

// #include <deal.II/lac/full_matrix.h>
// #include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <base/global.h>
#include <base/linear_algebra.h>
#include <function/factory.h>
#include <operator/stokes/matrix_based.h>

using namespace dealii;


namespace Operator
{
  namespace Stokes
  {
    template <int dim, typename LinearAlgebra, int spacedim>
    MatrixBased<dim, LinearAlgebra, spacedim>::MatrixBased(
      const hp::MappingCollection<dim, spacedim> &mapping_collection,
      const hp::QCollection<dim>                 &quadrature_collection,
      hp::FEValues<dim, spacedim>                &fe_values_collection)
      : mapping_collection(&mapping_collection)
      , quadrature_collection(&quadrature_collection)
      , fe_values_collection(&fe_values_collection)
    {}



    template <int dim, typename LinearAlgebra, int spacedim>
    void
    MatrixBased<dim, LinearAlgebra, spacedim>::reinit(
      const DoFHandler<dim, spacedim>     &dof_handler,
      const AffineConstraints<value_type> &constraints,
      VectorType                          &system_rhs)
    {
      TimerOutput::Scope t(getTimer(), "reinit");

      // ---
      // copied from setup_system()
      {
        mpi_communicator = dof_handler.get_communicator();

        std::vector<unsigned int> stokes_sub_blocks(dim + 1, 0);
        stokes_sub_blocks[dim] = 1;
        const std::vector<types::global_dof_index> dofs_per_block =
          DoFTools::count_dofs_per_fe_block(dof_handler, stokes_sub_blocks);

        const unsigned int n_u = dofs_per_block[0];
        const unsigned int n_p = dofs_per_block[1];

        owned_partitioning.resize(2);
        owned_partitioning[0] = dof_handler.locally_owned_dofs().get_view(0, n_u);
        owned_partitioning[1] =
          dof_handler.locally_owned_dofs().get_view(n_u, n_u + n_p);

        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
        relevant_partitioning.resize(2);
        relevant_partitioning[0] = locally_relevant_dofs.get_view(0, n_u);
        relevant_partitioning[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);
      }
      // ---

      system_matrix         = 0;
      preconditioner_matrix = 0;
      system_rhs            = 0;

      FullMatrix<double> cell_matrix;
      FullMatrix<double> cell_matrix2;
      Vector<double>     cell_rhs;

      // TODO: the RHS function is part of the problem class
      //       should be part of this class
      const std::unique_ptr<dealii::Function<dim>> rhs_function = Factory::create_function<dim>("kovasznay rhs");
      std::vector<Vector<double>> rhs_values;

      std::vector<Tensor<2, dim>> grad_phi_u;
      std::vector<double>         div_phi_u;
      std::vector<double>         phi_p;

      std::vector<types::global_dof_index> local_dof_indices;
      const FEValuesExtractors::Vector     velocities(0);
      const FEValuesExtractors::Scalar     pressure(dim);
      for (const auto &cell : dof_handler.active_cell_iterators() |
           IteratorFilters::LocallyOwnedCell())
        {
          fe_values_collection->reinit(cell);

          const FEValues<dim> &fe_values =
            fe_values_collection->get_present_fe_values();
          const unsigned int n_q_points = fe_values.n_quadrature_points;
          const unsigned int dofs_per_cell = fe_values.dofs_per_cell;

          cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
          cell_matrix = 0;
          cell_matrix2.reinit(dofs_per_cell, dofs_per_cell);
          cell_matrix2 = 0;
          cell_rhs.reinit(dofs_per_cell);
          cell_rhs = 0;

          grad_phi_u.resize(dofs_per_cell);
          div_phi_u.resize(dofs_per_cell);
          phi_p.resize(dofs_per_cell);

          local_dof_indices.resize(dofs_per_cell);

          // TODO: Move this part to the problem class???
          //       Not possible...
          rhs_values.resize(n_q_points, Vector<double>(dim + 1));
          rhs_function->vector_value_list(fe_values.get_quadrature_points(),
                                          rhs_values);

          // TODO: move to parameter
          const double viscosity = 0.1;

          for (unsigned int q_point = 0;
               q_point < fe_values.n_quadrature_points;
               ++q_point)
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
                        (viscosity *
                           scalar_product(grad_phi_u[i], grad_phi_u[j]) -
                         div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
                        fe_values.JxW(q_point);

                      cell_matrix2(i, j) += 1.0 / viscosity * phi_p[i] *
                                            phi_p[j] * fe_values.JxW(q_point);
                    }

                  const unsigned int component_i =
                    cell->get_fe().system_to_component_index(i).first;
                  cell_rhs(i) += fe_values.shape_value(i, q_point) *
                                 rhs_values[q_point](component_i) * fe_values.JxW(q_point);
                }
            }
          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);

          constraints.distribute_local_to_global(cell_matrix2,
                                                 local_dof_indices,
                                                 preconditioner_matrix);
        }

      system_rhs.compress(VectorOperation::values::add);
      system_matrix.compress(VectorOperation::values::add);
      preconditioner_matrix.compress(VectorOperation::add);
    }



    template <int dim, typename LinearAlgebra, int spacedim>
    void
    MatrixBased<dim, LinearAlgebra, spacedim>::vmult(
      VectorType       &dst,
      const VectorType &src) const
    {
      system_matrix.vmult(dst, src);
    }



    template <int dim, typename LinearAlgebra, int spacedim>
    void
    MatrixBased<dim, LinearAlgebra, spacedim>::initialize_dof_vector(
      VectorType &vec) const
    {
      if constexpr (std::is_same<typename LinearAlgebra::Vector,
                                 dealii::LinearAlgebra::distributed::Vector<
                                   double>>::value)
        {
          Assert(false, ExcNotImplemented());
        }
      else
        {
          vec.reinit(owned_partitioning,
                     relevant_partitioning,
                     mpi_communicator);
        }
    }



    template <int dim, typename LinearAlgebra, int spacedim>
    types::global_dof_index
    MatrixBased<dim, LinearAlgebra, spacedim>::m() const
    {
      return system_matrix.m();
    }



    template <int dim, typename LinearAlgebra, int spacedim>
    void
    MatrixBased<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(
      VectorType &diagonal) const
    {
      this->initialize_dof_vector(diagonal);

      for (auto entry : system_matrix)
        if (entry.row() == entry.column())
          diagonal[entry.row()] = 1.0 / entry.value();
    }



    template <int dim, typename LinearAlgebra, int spacedim>
    const typename LinearAlgebra::BlockSparseMatrix &
    MatrixBased<dim, LinearAlgebra, spacedim>::get_system_matrix() const
    {
      return system_matrix;
    }



    template <int dim, typename LinearAlgebra, int spacedim>
    void
    MatrixBased<dim, LinearAlgebra, spacedim>::Tvmult(
      VectorType       &dst,
      const VectorType &src) const
    {
      vmult(dst, src);
    }



// Do not build this class now.
/*
#ifdef DEAL_II_WITH_TRILINOS
    template class MatrixBased<2, dealiiTrilinos, 2>;
    template class MatrixBased<3, dealiiTrilinos, 3>;
    template class MatrixBased<2, Trilinos, 2>;
    template class MatrixBased<3, Trilinos, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
    template class MatrixBased<2, PETSc, 2>;
    template class MatrixBased<3, PETSc, 3>;
#endif
*/

  } // namespace Stokes
} // namespace Operator
