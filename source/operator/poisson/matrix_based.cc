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


#include <deal.II/dofs/dof_tools.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <operator/poisson/matrix_based.h>

using namespace dealii;

namespace Operator
{
  namespace Poisson
  {
    template <int dim, typename VectorType>
    MatrixBased<dim, VectorType>::MatrixBased(
      const hp::MappingCollection<dim> &   mapping,
      const DoFHandler<dim> &              dof_handler,
      const hp::QCollection<dim> &         quad,
      const AffineConstraints<value_type> &constraints,
      VectorType &                         system_rhs)
    {
      this->reinit(mapping, dof_handler, quad, constraints, system_rhs);
    }



    template <int dim, typename VectorType>
    void
    MatrixBased<dim, VectorType>::reinit(
      const hp::MappingCollection<dim> &   mapping_collection,
      const DoFHandler<dim> &              dof_handler,
      const hp::QCollection<dim> &         quadrature_collection,
      const AffineConstraints<value_type> &constraints,
      VectorType &                         system_rhs)
    {
      const MPI_Comm mpi_communicator = get_mpi_comm(dof_handler);

      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);

      this->partitioner = std::make_shared<const Utilities::MPI::Partitioner>(
        dof_handler.locally_owned_dofs(),
        locally_relevant_dofs,
        mpi_communicator);

      TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_dofs(),
                                            mpi_communicator);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      dsp.compress();

      system_matrix.reinit(dsp);

      initialize_dof_vector(system_rhs);

      // TODO: Create this object only once, i.e., in the constructor?
      hp::FEValues<dim> hp_fe_values(mapping_collection,
                                     dof_handler.get_fe_collection(),
                                     quadrature_collection,
                                     update_values | update_gradients |
                                       update_quadrature_points |
                                       update_JxW_values);

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
          hp_fe_values.reinit(cell);
          const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

          for (unsigned int q_point = 0;
               q_point < fe_values.n_quadrature_points;
               ++q_point)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q_point) * // grad phi_i(x_q)
                     fe_values.shape_grad(j, q_point) * // grad phi_j(x_q)
                     fe_values.JxW(q_point));           // dx
              }
          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }

      system_rhs.compress(VectorOperation::values::add);
      system_matrix.compress(VectorOperation::values::add);
    }



    template <int dim, typename VectorType>
    void
    MatrixBased<dim, VectorType>::vmult(VectorType &      dst,
                                        const VectorType &src) const
    {
      system_matrix.vmult(dst, src);
    }



    template <int dim, typename VectorType>
    void
    MatrixBased<dim, VectorType>::initialize_dof_vector(VectorType &vec) const
    {
      vec.reinit(partitioner);
    }



    template <int dim, typename VectorType>
    types::global_dof_index
    MatrixBased<dim, VectorType>::m() const
    {
      return system_matrix.m();
    }



    template <int dim, typename VectorType>
    void
    MatrixBased<dim, VectorType>::compute_inverse_diagonal(
      VectorType &diagonal) const
    {
      this->initialize_dof_vector(diagonal);

      for (auto entry : system_matrix)
        if (entry.row() == entry.column())
          diagonal[entry.row()] = 1.0 / entry.value();
    }



    template <int dim, typename VectorType>
    const TrilinosWrappers::SparseMatrix &
    MatrixBased<dim, VectorType>::get_system_matrix() const
    {
      return system_matrix;
    }



    template <int dim, typename VectorType>
    void
    MatrixBased<dim, VectorType>::Tvmult(VectorType &      dst,
                                         const VectorType &src) const
    {
      vmult(dst, src);
    }



    // explicit instantiations
    using VectorType = LinearAlgebra::distributed::Vector<double>;
    template class MatrixBased<2, VectorType>;
    template class MatrixBased<3, VectorType>;
  } // namespace Poisson
} // namespace Operator
