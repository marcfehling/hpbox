// ---------------------------------------------------------------------
//
// Copyright (C) 2023 - 2024 by the deal.II authors
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

#ifndef precondition_reduce_and_assemble_h
#define precondition_reduce_and_assemble_h


#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_pattern_base.h>

#include <deal.II/grid/filtered_iterator.h>


template <int dim, int spacedim, typename Number>
void
reduce_constraints(const dealii::DoFHandler<dim, spacedim>   &dof_handler,
                   const dealii::AffineConstraints<Number>   &constraints_full,
                   const std::vector<std::vector<dealii::types::global_dof_index>> &patch_indices,
                   const std::vector<std::vector<dealii::types::global_dof_index>> &patch_indices_ghost,
                   std::set<dealii::types::global_dof_index> &all_indices,
                   dealii::AffineConstraints<Number>         &constraints_reduced)
{
  Assert(constraints_full.is_closed(),
         dealii::ExcMessage("constraints_full needs to have all chains of constraints resolved"));
  Assert(constraints_reduced.get_locally_owned_indices().size() > 0,
         dealii::ExcMessage("constraints_reduced needs to be initialized"));

  // 1) create set of all patch indices
  all_indices.clear();

  for (const auto &indices : patch_indices)
    for (const auto &i : indices)
      all_indices.insert(i);

  for (const auto &indices : patch_indices_ghost)
    for (const auto &i : indices)
      all_indices.insert(i);

  // 2) store those indices that are constrained to the patch indices
  std::set<dealii::types::global_dof_index> constrained_indices;

  // ----------
  // TODO: find the right set for parallelization
  //       best guess: active cells
  //       move to parameter
  // ----------
  for (const auto i : dealii::DoFTools::extract_locally_active_dofs(dof_handler))
    if (constraints_full.is_constrained(i))
      {
        constrained_indices.insert(i);

        const auto constraint_entries =
          constraints_full.get_constraint_entries(i);

        std::vector<std::pair<dealii::types::global_dof_index, Number>>
          constraint_entries_reduced;

        if (constraint_entries != nullptr)
          for (const auto &entry : *constraint_entries)
            if (all_indices.contains(entry.first))
              constraint_entries_reduced.push_back(entry);

        constraints_reduced.add_line(i);
        constraints_reduced.add_entries(i, constraint_entries_reduced);
      }

  constraints_reduced.close();

  // add constrained indices to the set
  all_indices.merge(constrained_indices);
}



template <int dim, int spacedim, typename Number = double>
void
make_sparsity_pattern(const dealii::DoFHandler<dim, spacedim>         &dof_handler,
                      const std::set<dealii::types::global_dof_index> &all_indices,
                      dealii::SparsityPatternBase                     &sparsity_pattern,
                      const dealii::AffineConstraints<Number>         &constraints = {})
{
  std::vector<dealii::types::global_dof_index> local_dof_indices;
  std::vector<dealii::types::global_dof_index> local_dof_indices_reduced;

  const unsigned int max_dofs_per_cell = dof_handler.get_fe_collection().max_dofs_per_cell();
  local_dof_indices.reserve(max_dofs_per_cell);
  local_dof_indices_reduced.reserve(max_dofs_per_cell);

  // Note: there is a check for subdomain_id in DoFTools::make_sparsity_pattern,
  //       but we deemed it unnecessary here so we skipped it
  for (const auto &cell : dof_handler.active_cell_iterators() | dealii::IteratorFilters::LocallyOwnedCell())
    {
      const unsigned int n_dofs_per_cell = cell->get_fe().n_dofs_per_cell();
      local_dof_indices.resize(n_dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      local_dof_indices_reduced.clear();
      for (const auto i : local_dof_indices)
        if (all_indices.contains(i))
          local_dof_indices_reduced.push_back(i);

      constraints.add_entries_local_to_global(local_dof_indices_reduced,
                                              sparsity_pattern,
                                              /*keep_constrained_dofs=*/false);
    }
}



template <int dim, int spacedim, typename Number, typename SparseMatrixType>
void
partially_assemble_poisson(const dealii::DoFHandler<dim, spacedim>         &dof_handler,
                           const dealii::AffineConstraints<Number>         &constraints_reduced,
                           const dealii::hp::QCollection<dim>              &quadrature_collection,
                           const std::set<dealii::types::global_dof_index> &all_indices,
                           SparseMatrixType                                &sparse_matrix)
{
  //
  // build local matrices, distribute to sparse matrix
  //
  dealii::hp::FEValues<dim, spacedim> hp_fe_values(dof_handler.get_fe_collection(),
                                                   quadrature_collection,
                                                   dealii::update_gradients | dealii::update_JxW_values);

  dealii::FullMatrix<double>                   cell_matrix;
  std::vector<dealii::types::global_dof_index> local_dof_indices;

  // loop over locally owned cells
  for (const auto &cell : dof_handler.active_cell_iterators() | dealii::IteratorFilters::LocallyOwnedCell())
    {
      hp_fe_values.reinit(cell);

      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      std::vector<dealii::types::global_dof_index> local_dof_indices_reduced;
      std::vector<unsigned int>                    dof_indices;

      for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
        if (all_indices.contains(local_dof_indices[i]))
          {
            local_dof_indices_reduced.push_back(local_dof_indices[i]);
            dof_indices.push_back(i);
          }

      if (dof_indices.empty())
        continue;

      const auto &fe_values = hp_fe_values.get_present_fe_values();

      cell_matrix.reinit(dof_indices.size(), dof_indices.size());

      // loop over cell dofs
      for (const auto q : fe_values.quadrature_point_indices())
        {
          for (unsigned int i = 0; i < dof_indices.size(); ++i)
            for (unsigned int j = 0; j < dof_indices.size(); ++j)
              cell_matrix(i, j) +=
                (fe_values.shape_grad(dof_indices[i], q) * // grad phi_i(x_q)
                 fe_values.shape_grad(dof_indices[j], q) * // grad phi_j(x_q)
                 fe_values.JxW(q));                        // dx
        }

      constraints_reduced.distribute_local_to_global(cell_matrix,
                                                     local_dof_indices_reduced,
                                                     sparse_matrix);
    }

  sparse_matrix.compress(dealii::VectorOperation::values::add);
}



template <int dim, int spacedim, typename Number, typename SparseMatrixType>
void
partially_assemble_ablock(const dealii::DoFHandler<dim, spacedim>         &dof_handler,
                          const dealii::AffineConstraints<Number>         &constraints_reduced,
                          const dealii::hp::QCollection<dim>              &quadrature_collection,
                          const std::set<dealii::types::global_dof_index> &all_indices,
                          SparseMatrixType                                &sparse_matrix)
{
  //
  // build local matrices, distribute to sparse matrix
  //

  // TODO: take fe values from somewhere else?

  dealii::hp::FEValues<dim, spacedim> hp_fe_values(dof_handler.get_fe_collection(),
                                                   quadrature_collection,
                                                   dealii::update_gradients | dealii::update_JxW_values);

  dealii::FullMatrix<double>                   cell_matrix;
  std::vector<dealii::Tensor<2, dim>>          grad_phi_u;
  std::vector<dealii::types::global_dof_index> local_dof_indices;

  // loop over locally owned cells
  for (const auto &cell : dof_handler.active_cell_iterators() | dealii::IteratorFilters::LocallyOwnedCell())
    {
      hp_fe_values.reinit(cell);

      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      std::vector<dealii::types::global_dof_index> local_dof_indices_reduced;
      std::vector<unsigned int>                    dof_indices;

      for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
        if (all_indices.contains(local_dof_indices[i]))
          {
            local_dof_indices_reduced.push_back(local_dof_indices[i]);
            dof_indices.push_back(i);
          }

      if (dof_indices.empty())
        continue;

      const auto &fe_values = hp_fe_values.get_present_fe_values();

      cell_matrix.reinit(dof_indices.size(), dof_indices.size());
      grad_phi_u.resize(dof_indices.size());

      // TODO: move to parameter
      const double viscosity = 0.1;

      // loop over cell dofs
      const dealii::FEValuesExtractors::Vector velocities(0);
      for (const auto q : fe_values.quadrature_point_indices())
        {
          for (unsigned int k = 0; k < dof_indices.size(); ++k)
            grad_phi_u[k] = fe_values[velocities].gradient(dof_indices[k], q);

          for (unsigned int i = 0; i < dof_indices.size(); ++i)
            for (unsigned int j = 0; j < dof_indices.size(); ++j)
              cell_matrix(i, j) += viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]) *
                                   fe_values.JxW(q);
        }

      constraints_reduced.distribute_local_to_global(cell_matrix,
                                                     local_dof_indices_reduced,
                                                     sparse_matrix);
    }

  sparse_matrix.compress(dealii::VectorOperation::values::add);
}


#endif
