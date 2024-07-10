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

#ifndef multigrid_reduce_and_assemble_h
#define multigrid_reduce_and_assemble_h


#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparsity_pattern_base.h>

#include <deal.II/matrix_free/fe_evaluation.h>


inline std::set<dealii::types::global_dof_index>
extract_relevant(const std::vector<std::vector<dealii::types::global_dof_index>> &patch_indices,
                 const Partitioning                                              &partitioning)
{
  std::set<dealii::types::global_dof_index> all_indices_relevant;

  // ----------
  // TODO: is there a better way to do that?
  //       via GridTools::exchange_cell_data_to_ghosts
  dealii::LinearAlgebra::distributed::Vector<float> count_patch_dofs(
    partitioning.get_owned_dofs(),
    partitioning.get_relevant_dofs(),
    partitioning.get_communicator());

  for (const auto &indices : patch_indices)
    for (const auto i : indices)
      ++count_patch_dofs[i];

  count_patch_dofs.compress(dealii::VectorOperation::add);
  count_patch_dofs.update_ghost_values();

  for (const auto i : partitioning.get_relevant_dofs())
    if (count_patch_dofs[i] > 0.)
      all_indices_relevant.insert(i);
  // ----------

  // all_indices_relevant now contains locally relevant dofs that are patch dofs

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
         dealii::ExcMessage("constraints_full needs to have all chains of constraints "
                            "and boundary constraints resolved."));

  // 1) reduce constraints

  // store those locally active indices that are constrained
  std::set<dealii::types::global_dof_index> all_indices_constrained;

  // extract constraints of locally active dofs against any locally relevant patch index
  for (const auto i : locally_active_dofs)
    if (constraints_full.is_constrained(i))
      {
        const auto constraint_entries = constraints_full.get_constraint_entries(i);

        std::vector<std::pair<dealii::types::global_dof_index, Number>> constraint_entries_reduced;

        if (constraint_entries != nullptr)
          for (const auto &entry : *constraint_entries)
            // TODO C++20: use std::set::contains instead
            if (all_indices_relevant.find(entry.first) != all_indices_relevant.end())
              constraint_entries_reduced.push_back(entry);

        if (constraint_entries_reduced.empty() == false)
          {
            all_indices_constrained.insert(i);

            constraints_reduced.add_line(i);
            constraints_reduced.add_entries(i, constraint_entries_reduced);
          }
      }

  constraints_reduced.close();


  // 2) to assemble sparse matrices partially only on patch indices, we will create the set of
  // indices necessary in all_indices_assemble.
  all_indices_assemble.clear();

  // first, we need all patch indices on locally active cells
  std::set_intersection(all_indices_relevant.begin(),
                        all_indices_relevant.end(),
                        locally_active_dofs.begin(),
                        locally_active_dofs.end(),
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
  for (const auto &cell :
       dof_handler.active_cell_iterators() | dealii::IteratorFilters::LocallyOwnedCell())
    {
      const unsigned int n_dofs_per_cell = cell->get_fe().n_dofs_per_cell();
      local_dof_indices.resize(n_dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      local_dof_indices_reduced.clear();
      for (const auto i : local_dof_indices)
        // TODO C++20: use std::set::contains instead
        if (all_indices_assemble.find(i) != all_indices_assemble.end())
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
                           const dealii::hp::QCollection<dim> &,
                           const std::set<dealii::types::global_dof_index> &all_indices_assemble,
                           SparseMatrixType                                &sparse_matrix)
{
  //
  // build local matrices, distribute to sparse matrix
  //
  const auto &fes = dof_handler.get_fe_collection();
  std::vector<std::unique_ptr<dealii::FEEvaluation<dim, -1, 0, dim, double>>> evaluators(
    fes.size());
  for (unsigned int i = 0; i < fes.size(); ++i)
    evaluators[i] = std::make_unique<dealii::FEEvaluation<dim, -1, 0, dim, double>>(
      fes[i],
      dealii::QGauss<1>(fes[i].degree + 1),
      dealii::update_gradients | dealii::update_JxW_values);

  dealii::FullMatrix<double>                   cell_matrix;
  std::vector<dealii::types::global_dof_index> local_dof_indices;

  // loop over locally owned cells
  for (const auto &cell :
       dof_handler.active_cell_iterators() | dealii::IteratorFilters::LocallyOwnedCell())
    {
      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      std::vector<dealii::types::global_dof_index> local_dof_indices_reduced;
      std::vector<unsigned int>                    dof_indices;
      auto                                        &evaluator = *evaluators[cell->active_fe_index()];
      const std::vector<unsigned int>             &lexicographic =
        evaluator.get_shape_info().lexicographic_numbering;

      for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
        // TODO C++20: use std::set::contains instead
        if (all_indices_assemble.find(local_dof_indices[lexicographic[i]]) !=
            all_indices_assemble.end())
          {
            local_dof_indices_reduced.push_back(local_dof_indices[lexicographic[i]]);
            dof_indices.push_back(i);
          }

      if (dof_indices.empty())
        continue;

      cell_matrix.reinit(dof_indices.size(), dof_indices.size());
      evaluator.reinit(cell);

      // loop over cell dofs
      constexpr unsigned int n_lanes = dealii::VectorizedArray<double>::size();
      for (unsigned int k = 0; k < dof_indices.size(); k += n_lanes)
        {
          dealii::VectorizedArray<double> *dof_values = evaluator.begin_dof_values();
          for (unsigned int i = 0; i < evaluator.dofs_per_cell; ++i)
            dof_values[i] = {};
          for (unsigned int j = k; j < dof_indices.size() && j - k < n_lanes; ++j)
            dof_values[dof_indices[j]][j - k] = 1.0;

          evaluator.evaluate(dealii::EvaluationFlags::gradients);
          for (unsigned int q = 0; q < evaluator.n_q_points; ++q)
            evaluator.submit_gradient(evaluator.get_gradient(q), q);
          evaluator.integrate(dealii::EvaluationFlags::gradients);

          for (unsigned int j = k; j < dof_indices.size() && j - k < n_lanes; ++j)
            for (unsigned int i = 0; i < dof_indices.size(); ++i)
              cell_matrix(i, j) = dof_values[dof_indices[i]][j - k];
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
                          const dealii::hp::QCollection<dim> &,
                          const std::set<dealii::types::global_dof_index> &all_indices_assemble,
                          SparseMatrixType                                &sparse_matrix)
{
  //
  // build local matrices, distribute to sparse matrix
  //

  const auto &fes = dof_handler.get_fe_collection();
  std::vector<std::unique_ptr<dealii::FEEvaluation<dim, -1, 0, dim, double>>> evaluators(
    fes.size());
  for (unsigned int i = 0; i < fes.size(); ++i)
    evaluators[i] = std::make_unique<dealii::FEEvaluation<dim, -1, 0, dim, double>>(
      fes[i],
      dealii::QGauss<1>(fes[i].degree + 1),
      dealii::update_gradients | dealii::update_JxW_values);

  dealii::FullMatrix<double>                   cell_matrix;
  std::vector<dealii::types::global_dof_index> local_dof_indices;

  // loop over locally owned cells
  for (const auto &cell :
       dof_handler.active_cell_iterators() | dealii::IteratorFilters::LocallyOwnedCell())
    {
      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      std::vector<dealii::types::global_dof_index> local_dof_indices_reduced;
      std::vector<unsigned int>                    dof_indices;
      auto                                        &evaluator = *evaluators[cell->active_fe_index()];
      const std::vector<unsigned int>             &lexicographic =
        evaluator.get_shape_info().lexicographic_numbering;

      for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
        // TODO C++20: use std::set::contains instead
        if (all_indices_assemble.find(local_dof_indices[lexicographic[i]]) !=
            all_indices_assemble.end())
          {
            local_dof_indices_reduced.push_back(local_dof_indices[lexicographic[i]]);
            dof_indices.push_back(i);
          }

      if (dof_indices.empty())
        continue;

      cell_matrix.reinit(dof_indices.size(), dof_indices.size());
      evaluator.reinit(cell);

      // TODO: move to parameter
      const double viscosity = 0.1;

      // loop over cell dofs
      constexpr unsigned int n_lanes = dealii::VectorizedArray<double>::size();
      for (unsigned int k = 0; k < dof_indices.size(); k += n_lanes)
        {
          dealii::VectorizedArray<double> *dof_values = evaluator.begin_dof_values();
          for (unsigned int i = 0; i < evaluator.dofs_per_cell; ++i)
            dof_values[i] = {};
          for (unsigned int j = k; j < dof_indices.size() && j - k < n_lanes; ++j)
            dof_values[dof_indices[j]][j - k] = 1.0;

          evaluator.evaluate(dealii::EvaluationFlags::gradients);
          for (unsigned int q = 0; q < evaluator.n_q_points; ++q)
            evaluator.submit_gradient(dealii::make_vectorized_array(viscosity) *
                                        evaluator.get_gradient(q),
                                      q);
          evaluator.integrate(dealii::EvaluationFlags::gradients);

          for (unsigned int j = k; j < dof_indices.size() && j - k < n_lanes; ++j)
            for (unsigned int i = 0; i < dof_indices.size(); ++i)
              cell_matrix(i, j) = dof_values[dof_indices[i]][j - k];
        }

      constraints_reduced.distribute_local_to_global(cell_matrix,
                                                     local_dof_indices_reduced,
                                                     sparse_matrix);
    }

  sparse_matrix.compress(dealii::VectorOperation::values::add);
}


#endif
