// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2022 by the deal.II authors
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

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <base/global.h>
#include <base/linear_algebra.h>
#include <poisson/matrixbased.h>

using namespace dealii;


namespace Poisson
{
  template <int dim, typename LinearAlgebra, int spacedim>
  OperatorMatrixBased<dim, LinearAlgebra, spacedim>::OperatorMatrixBased(
    const hp::MappingCollection<dim, spacedim> &mapping_collection,
    const hp::QCollection<dim>                 &quadrature_collection,
    hp::FEValues<dim, spacedim>                &fe_values_collection)
    : mapping_collection(&mapping_collection)
    , quadrature_collection(&quadrature_collection)
    , fe_values_collection(&fe_values_collection)
  {}



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  OperatorMatrixBased<dim, LinearAlgebra, spacedim>::reinit(
    const dealii::DoFHandler<dim, spacedim>     &dof_handler,
    const dealii::AffineConstraints<value_type> &constraints,
    VectorType                                  &system_rhs)
  {
    {
      TimerOutput::Scope t(getTimer(), "setup_system");

      const MPI_Comm mpi_communicator = dof_handler.get_communicator();

      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);

      this->partitioner = std::make_shared<const Utilities::MPI::Partitioner>(
        dof_handler.locally_owned_dofs(),
        locally_relevant_dofs,
        mpi_communicator);

      // constructors differ, so we need this workaround... -- not happy with
      // this
      if constexpr (std::is_same<LinearAlgebra, PETSc>::value)
        {
          TimerOutput::Scope t(getTimer(), "reinit_matrix");

          typename LinearAlgebra::SparsityPattern dsp(locally_relevant_dofs);

          {
            TimerOutput::Scope t(getTimer(), "make_sparsity_pattern");

            DoFTools::make_sparsity_pattern(dof_handler,
                                            dsp,
                                            constraints,
                                            false);
          }

          SparsityTools::distribute_sparsity_pattern(
            dsp,
            dof_handler.locally_owned_dofs(),
            mpi_communicator,
            locally_relevant_dofs);

          system_matrix.reinit(dof_handler.locally_owned_dofs(),
                               dof_handler.locally_owned_dofs(),
                               dsp,
                               mpi_communicator);
        }
      else if constexpr (std::is_same<LinearAlgebra, Trilinos>::value ||
                         std::is_same<LinearAlgebra, dealiiTrilinos>::value)
        {
          TimerOutput::Scope t(getTimer(), "reinit_matrix");

          typename LinearAlgebra::SparsityPattern dsp(
            dof_handler.locally_owned_dofs(), mpi_communicator);

          {
            TimerOutput::Scope t(getTimer(), "make_sparsity_pattern");

            DoFTools::make_sparsity_pattern(dof_handler,
                                            dsp,
                                            constraints,
                                            false);
          }

          dsp.compress();

          system_matrix.reinit(dsp);
        }
      else
        {
          Assert(false, ExcNotImplemented());
        }

      {
        TimerOutput::Scope(getTimer(), "reinit_vectors");

        system_rhs.reinit(partitioner->locally_owned_range(),
                          partitioner->get_mpi_communicator());
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

          fe_values_collection->reinit(cell);
          const FEValues<dim> &fe_values =
            fe_values_collection->get_present_fe_values();

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
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  OperatorMatrixBased<dim, LinearAlgebra, spacedim>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    system_matrix.vmult(dst, src);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  OperatorMatrixBased<dim, LinearAlgebra, spacedim>::initialize_dof_vector(
    VectorType &vec) const
  {
    if constexpr (std::is_same<
                    typename LinearAlgebra::Vector,
                    dealii::LinearAlgebra::distributed::Vector<double>>::value)
      {
        vec.reinit(partitioner);
      }
    else
      {
        vec.reinit(partitioner->locally_owned_range(),
                   partitioner->ghost_indices(),
                   partitioner->get_mpi_communicator());
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  types::global_dof_index
  OperatorMatrixBased<dim, LinearAlgebra, spacedim>::m() const
  {
    return system_matrix.m();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  OperatorMatrixBased<dim, LinearAlgebra, spacedim>::compute_inverse_diagonal(
    VectorType &diagonal) const
  {
    this->initialize_dof_vector(diagonal);

    for (auto entry : system_matrix)
      if (entry.row() == entry.column())
        diagonal[entry.row()] = 1.0 / entry.value();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  const typename LinearAlgebra::SparseMatrix &
  OperatorMatrixBased<dim, LinearAlgebra, spacedim>::get_system_matrix() const
  {
    return system_matrix;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  OperatorMatrixBased<dim, LinearAlgebra, spacedim>::Tvmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    vmult(dst, src);
  }



// explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class OperatorMatrixBased<2, dealiiTrilinos, 2>;
  template class OperatorMatrixBased<3, dealiiTrilinos, 3>;
  template class OperatorMatrixBased<2, Trilinos, 2>;
  template class OperatorMatrixBased<3, Trilinos, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
  template class OperatorMatrixBased<2, PETSc, 2>;
  template class OperatorMatrixBased<3, PETSc, 3>;
#endif

} // namespace Poisson
