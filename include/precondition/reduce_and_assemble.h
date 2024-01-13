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


template <int dim, int spacedim>
std::set<dealii::types::global_dof_index>
extract_relevant(const dealii::DoFHandler<dim, spacedim>                         &dof_handler,
                 const std::vector<std::vector<dealii::types::global_dof_index>> &patch_indices)
{
  std::set<dealii::types::global_dof_index> all_indices_relevant;

  // ----------
  // TODO: is there a better way to do that?
  //       via GridTools::exchange_cell_data_to_ghosts
  const dealii::IndexSet &owned = dof_handler.locally_owned_dofs();
  const dealii::IndexSet  relevant = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

  dealii::LinearAlgebra::distributed::Vector<float> count_patch_dofs (owned, relevant, dof_handler.get_communicator());
  for (const auto& indices : patch_indices)
    for (const auto i : indices)
      ++count_patch_dofs[i];

  count_patch_dofs.compress(dealii::VectorOperation::add);
  count_patch_dofs.update_ghost_values();

  for (const auto i : relevant)
    if (count_patch_dofs[i] > 0.)
      all_indices_relevant.insert(i);
  // ----------

  // relevant now contains locally relevant dofs that are patch dofs

  return all_indices_relevant;
}



template <typename Number>
void
reduce_constraints(const dealii::AffineConstraints<Number>         &constraints_full,
                   const dealii::IndexSet                          &locally_active_dofs,
                   const std::set<dealii::types::global_dof_index> &all_indices_relevant,
                   dealii::AffineConstraints<Number>               &constraints_reduced,
                   std::set<dealii::types::global_dof_index>       &all_indices_assemble)
{
  Assert(constraints_full.is_closed(),
         dealii::ExcMessage("constraints_full needs to have all chains of constraints resolved"));

  // 1) reduce constraints

  // store those locally active indices that are constrained
  // TODO: use constraints_reduced.get_local_lines() instead
  //       of building all_indices_constrained manually?
  std::set<dealii::types::global_dof_index> all_indices_constrained;

  // extract constraints of locally active dofs against any locally relevant patch index
  for (const auto i : locally_active_dofs)
    if (constraints_full.is_constrained(i))
      {
        const auto constraint_entries =
          constraints_full.get_constraint_entries(i);

        std::vector<std::pair<dealii::types::global_dof_index, Number>>
          constraint_entries_reduced;

        if (constraint_entries != nullptr)
          for (const auto &entry : *constraint_entries)
            if (all_indices_relevant.contains(entry.first))
              constraint_entries_reduced.push_back(entry);

        if (constraint_entries_reduced.empty() == false)
          {
            // add entries if index is constrained against any patch index
            all_indices_constrained.insert(i);

            constraints_reduced.add_line(i);
            constraints_reduced.add_entries(i, constraint_entries_reduced);
          }
        else if (all_indices_relevant.contains(i))
          {
            // set constrained patch index to zero
            constraints_reduced.add_line(i);

            std::cout << "encountered constrained patch dof" << std::endl;
          }
      }

  constraints_reduced.close();


  // 2) to assemble sparse matrices partially only on patch indices, we will create the set of indices necessary in all_indices_assemble.
  all_indices_assemble.clear();

  // first, we need all patch indices on locally active cells
  std::set_intersection(all_indices_relevant.begin(), all_indices_relevant.end(),
                        locally_active_dofs.begin(), locally_active_dofs.end(),
                        std::inserter(all_indices_assemble, all_indices_assemble.begin()));

  // second, we need those locally active indices that are constrained
  all_indices_assemble.merge(all_indices_constrained);
}



template <int dim, int spacedim, typename Number = double>
void
make_sparsity_pattern(const dealii::DoFHandler<dim, spacedim>         &dof_handler,
                      const std::set<dealii::types::global_dof_index> &all_indices_assemble,
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
        if (all_indices_assemble.contains(i))
          local_dof_indices_reduced.push_back(i);

      constraints.add_entries_local_to_global(local_dof_indices_reduced,
                                              sparsity_pattern,
                                              /*keep_constrained_dofs=*/false);
    }
}



template <int dim, int spacedim, typename Number, typename SparseMatrixType>
void
partially_assemble_poisson(const dealii::DoFHandler<dim, spacedim> &dof_handler,
                           const dealii::AffineConstraints<Number> &constraints_reduced,
                           const dealii::hp::QCollection<dim>      &quadrature_collection,
                           const std::set<dealii::types::global_dof_index> &all_indices_assemble,
                           SparseMatrixType                        &sparse_matrix)
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
        if (all_indices_assemble.contains(local_dof_indices[i]))
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
partially_assemble_ablock(const dealii::DoFHandler<dim, spacedim> &dof_handler,
                          const dealii::AffineConstraints<Number> &constraints_reduced,
                          const dealii::hp::QCollection<dim>      &quadrature_collection,
                          const std::set<dealii::types::global_dof_index> &all_indices_assemble,
                          SparseMatrixType                        &sparse_matrix)
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
        if (all_indices_assemble.contains(local_dof_indices[i]))
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
