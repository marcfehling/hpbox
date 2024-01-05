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

#ifndef precondition_diag_ext_h
#define precondition_diag_ext_h


#include <deal.II/base/mpi.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/sparse_matrix_tools.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.templates.h>

#include <global.h>

DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim = dim>
void
prepare_patch_indices(const DoFHandler<dim, spacedim> &dof_handler,
                      const AffineConstraints<double> &constraints,
                      std::vector<std::vector<types::global_dof_index>> &patch_indices,
                      std::vector<std::vector<types::global_dof_index>> &patch_indices_ghost)
{
  patch_indices.clear();
  patch_indices_ghost.clear();

  std::vector<types::global_dof_index> local_indices;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() || cell->is_ghost())
      for (const auto f : cell->face_indices())
        if (cell->at_boundary(f) == false)
          {
            bool flag = false;

            if (cell->face(f)->has_children())
              for (unsigned int sf = 0;
                    sf < cell->face(f)->n_children();
                    ++sf)
                {
                  const auto neighbor_subface =
                    cell->neighbor_child_on_subface(f, sf);

                  // check faces among locally owned and on interfaces with ghost cells
                  // to cover all patches that possibly contain locally active dofs
                  if (neighbor_subface->is_locally_owned() || neighbor_subface->is_ghost())
                    // problem criterion: cell faces h-refined cell with lower polynomial degree
                    if (neighbor_subface->get_fe().degree < cell->get_fe().degree)
                      {
                        flag = true;
                        break;
                      }
                }

            if (flag == false)
              continue;

            local_indices.resize(cell->get_fe().n_dofs_per_face());
            cell->face(f)->get_dof_indices(local_indices,
                                          cell->active_fe_index());

            std::vector<types::global_dof_index> local_unconstrained_indices;
            for (const auto i : local_indices)
              if (constraints.is_constrained(i) == false)
                local_unconstrained_indices.emplace_back(i);

            if (local_unconstrained_indices.empty() == false)
              {
                if (cell->is_locally_owned())
                  patch_indices.push_back(std::move(local_unconstrained_indices));
                else // ghost cell
                  patch_indices_ghost.push_back(std::move(local_unconstrained_indices));
              }
          }
}



template <int dim, typename Number, int spacedim = dim>
void
reduce_constraints(const DoFHandler<dim, spacedim>   &dof_handler,
                   const AffineConstraints<Number>   &constraints_full,
                   const std::vector<std::vector<types::global_dof_index>> &patch_indices,
                   const std::vector<std::vector<types::global_dof_index>> &patch_indices_ghost,
                   std::set<types::global_dof_index> &all_indices,
                   AffineConstraints<Number>         &constraints_reduced)
{
  Assert(constraints_full.is_closed(),
         ExcMessage("constraints_full needs to have all chains of constraints resolved"));

  // create set of all patch indices
  all_indices.clear();

  for (const auto &indices : patch_indices)
    for (const auto &i : indices)
      all_indices.insert(i);

  for (const auto &indices : patch_indices_ghost)
    for (const auto &i : indices)
      all_indices.insert(i);

  // store those indices that are constrained to the patch indices
  std::set<types::global_dof_index> constrained_indices;

  const auto &owned    = dof_handler.locally_owned_dofs();
  const auto  active   = DoFTools::extract_locally_active_dofs(dof_handler);
  const auto  relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // reduce constraints to those that affect patch indices
  // ----------
  // TODO: move initialization outside of this function?
  // ----------
  constraints_reduced.reinit(owned, relevant);

  // ----------
  // TODO: find the right set for parallelization
  //       best guess: active cells
  // ----------
  for (const auto i : active)
    if (constraints_full.is_constrained(i))
      {
        constrained_indices.insert(i);

        const auto constraint_entries =
          constraints_full.get_constraint_entries(i);

        std::vector<std::pair<types::global_dof_index, Number>>
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
make_sparsity_pattern(const DoFHandler<dim, spacedim>         &dof_handler,
                      const std::set<types::global_dof_index> &all_indices,
                      SparsityPatternBase                     &sparsity_pattern,
                      const AffineConstraints<Number>         &constraints = {})
{
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<types::global_dof_index> local_dof_indices_reduced;

  const unsigned int max_dofs_per_cell = dof_handler.get_fe_collection().max_dofs_per_cell();
  local_dof_indices.reserve(max_dofs_per_cell);
  local_dof_indices_reduced.reserve(max_dofs_per_cell);

  // Note: there is a check for subdomain_id in DoFTools::make_sparsity_pattern,
  //       but we deemed it unnecessary here so we skipped it
  for (const auto &cell : dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
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


// ----------
// TODO: change the interface to:
//       (dof_handler, constraints, quadrature, all_indices, sparse_matrix)
//       i.e. get rid of patch_indices and sparsity_pattern
// ----------
template <int dim, typename Number, typename SparseMatrixType, typename SparsityPatternType, int spacedim = dim>
void
partial_assembly_poisson(const DoFHandler<dim, spacedim> &dof_handler,
                         const AffineConstraints<Number> &constraints_full,
                         const hp::QCollection<dim>      &quadrature_collection,
                         const std::vector<std::vector<types::global_dof_index>> &patch_indices,
                         const std::vector<std::vector<types::global_dof_index>> &patch_indices_ghost,
                         SparseMatrixType                &sparse_matrix,
                         SparsityPatternType             &sparsity_pattern)
{
  // ----------
  // TODO: figure where to move this stuff
  //       use Partitioning class???
  // ----------

  const auto &owned    = dof_handler.locally_owned_dofs();
  const auto  active   = DoFTools::extract_locally_active_dofs(dof_handler);
  const auto  relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);

  AffineConstraints<Number>         constraints;
  std::set<types::global_dof_index> all_indices;

  reduce_constraints(dof_handler, constraints_full, patch_indices, patch_indices_ghost,
                     all_indices, constraints);


  //
  // create sparsity pattern on reduced constraints
  //
  // TODO: This works only for TrilinosWrappers::SparsityPattern
  sparsity_pattern.reinit(owned, owned, relevant, dof_handler.get_communicator());

  make_sparsity_pattern(dof_handler, all_indices, sparsity_pattern, constraints);

  sparsity_pattern.compress();

  sparse_matrix.reinit(sparsity_pattern);

  //
  // build local matrices, distribute to sparse matrix
  //
  // TODO: make this the beginning of the 'partial assembly' functions
  hp::FEValues<dim> hp_fe_values(dof_handler.get_fe_collection(),
                                 quadrature_collection,
                                 update_gradients | update_JxW_values);

  FullMatrix<double>                   cell_matrix;
  std::vector<types::global_dof_index> local_dof_indices;

  // loop over locally owned cells
  for (const auto &cell : dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
    {
      hp_fe_values.reinit(cell);

      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      std::vector<types::global_dof_index> local_dof_indices_reduced;
      std::vector<unsigned int>            dof_indices;

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

      constraints.distribute_local_to_global(cell_matrix,
                                             local_dof_indices_reduced,
                                             sparse_matrix);
    }

  sparse_matrix.compress(VectorOperation::values::add);
}



template<typename VectorType, int dim, int spacedim>
class ExtendedDiagonalPreconditioner
{
private:
  enum class WeightingType
  {
    none,
    left,
    right,
    symm
  };

  using Number = typename VectorType::value_type;

public:
  ExtendedDiagonalPreconditioner(const DoFHandler<dim, spacedim>                         &dof_handler,
                                 const std::vector<std::vector<types::global_dof_index>> &patch_indices)
    : dof_handler(dof_handler)
    , patch_indices(patch_indices)
    , weighting_type(WeightingType::symm)
  {}

  template <typename GlobalSparseMatrixType, typename GlobalSparsityPattern>
  void
  initialize(const GlobalSparseMatrixType &global_sparse_matrix,
             const GlobalSparsityPattern  &global_sparsity_pattern,
             const VectorType             &inverse_diagonal)
  {
    TimerOutput::Scope t(getTimer(), "initialize_extdiag");

    // we need a partitioner over locally relevant dofs,
    // as patch dofs might be constrained with ghost dofs
    // const auto relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
    // const auto large_partitioner =
    //   std::make_shared<const Utilities::MPI::Partitioner>(
    //     dof_handler.locally_owned_dofs(),
    //     relevant_dofs,
    //     dof_handler.get_communicator());
    const auto large_partitioner = inverse_diagonal.get_partitioner();

    //
    // build patch matrices
    //
    SparseMatrixTools::restrict_to_full_matrices(global_sparse_matrix,
                                                 global_sparsity_pattern,
                                                 patch_indices,
                                                 patch_matrices);

    for (auto &block : patch_matrices)
      block.gauss_jordan();

    //
    // prepare weights
    //
    VectorType     weights;
    Vector<Number> vector_weights;
    if (weighting_type != WeightingType::none)
      {
        weights.reinit(large_partitioner);

        for (unsigned int c = 0; c < patch_indices.size(); ++c)
          {
            const unsigned int dofs_per_cell = patch_indices[c].size();
            vector_weights.reinit(dofs_per_cell);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              vector_weights[i] = 1.0;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              weights[patch_indices[c][i]] += vector_weights[i];
          }

        weights.compress(VectorOperation::add);
        for (auto &i : weights)
          i = (weighting_type == WeightingType::symm) ? std::sqrt(1.0 / i) :
                                                        (1.0 / i);
        weights.update_ghost_values();
      }

    //
    // multiply weights to patch matrices (blocks)
    //
    if (weighting_type != WeightingType::none)
      {
        for (unsigned int cell = 0; cell < patch_indices.size(); ++cell)
          {
            const unsigned int dofs_per_cell = patch_indices[cell].size();

            vector_weights.reinit(dofs_per_cell);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              vector_weights[i] += weights[patch_indices[cell][i]];

            auto & block = patch_matrices[cell];

            if (weighting_type == WeightingType::symm ||
                weighting_type == WeightingType::right)
              {
                // multiply weights from right B(wI), i.e.,
                // multiply one weight for each column
                for (unsigned int r = 0; r < dofs_per_cell; ++r)
                  for (unsigned int c = 0; c < dofs_per_cell; ++c)
                    block(r,c) *= vector_weights[c];
              }

            if (weighting_type == WeightingType::symm ||
                weighting_type == WeightingType::left)
              {
                // multiply weights from left (wI)B, i.e.,
                // multiply one weight for each row
                for (unsigned int r = 0; r < dofs_per_cell; ++r)
                  for (unsigned int c = 0; c < dofs_per_cell; ++c)
                    block(r,c) *= vector_weights[r];
              }
          }
      }

    //
    // clear diagonal entries assigned to an ASM patch
    //
    // first, count how often indices occur in patches
    // TODO: this is suboptimal, we can avoid the ghost exchange
    //       as we have all information already with prepare_patch_indices,
    //       see below
    VectorType unprocessed_indices(large_partitioner);
    for (const auto &indices_i : patch_indices)
      for (const auto i : indices_i)
        unprocessed_indices[i]++;

    unprocessed_indices.compress(VectorOperation::add);
    unprocessed_indices.update_ghost_values();

    this->inverse_diagonal = inverse_diagonal;
    for (const auto l : large_partitioner->locally_owned_range())
      if (unprocessed_indices[l] > 0)
        this->inverse_diagonal[l] = 0.0;

    std::vector<types::global_dof_index> ghost_indices;
    for (const auto g : large_partitioner->ghost_indices())
      if (unprocessed_indices[g] > 0)
        ghost_indices.push_back(g);

    // TODO: i tried this approach, but we need info about patches
    //       between ghost cells
    // this->inverse_diagonal = inverse_diagonal;
    // std::vector<types::global_dof_index> ghost_indices;
    // for (const auto &indices : patch_indices)
    //   for (const auto i : indices)
    //     {
    //       if (large_partitioner->in_local_range(i))
    //         this->inverse_diagonal[i] = 0.0;
    //       else
    //         ghost_indices.push_back(i);
    //     }
    //
    // for (const auto &indices : patch_indices_ghost)
    //  for (const auto i : indices)
    //    if (large_partitioner->in_ghost_range(i))
    //      ghost_indices.push_back(i);


    //
    // set embedded partitioner
    //
    std::sort(ghost_indices.begin(), ghost_indices.end());
    ghost_indices.erase(std::unique(ghost_indices.begin(), ghost_indices.end()),
                        ghost_indices.end());

    IndexSet ghost_indices_is(large_partitioner->size());
    ghost_indices_is.add_indices(ghost_indices.begin(), ghost_indices.end());

    Assert(ghost_indices_is.is_subset_of(large_partitioner->ghost_indices()),
           ExcMessage("Ghost range mismatch!"));

    const auto partitioner =
      std::make_shared<const Utilities::MPI::Partitioner>(
        large_partitioner->locally_owned_range(),
        ghost_indices_is,
        large_partitioner->get_mpi_communicator());

    this->embedded_partitioner =
      internal::create_embedded_partitioner(partitioner, large_partitioner);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    TimerOutput::Scope t(getTimer(), "vmult_extdiag");

    // apply inverse diagonal
    internal::DiagonalMatrix::assign_and_scale(dst, src, this->inverse_diagonal);

    // apply ASM: 1) update ghost values
    internal::SimpleVectorDataExchange<Number> data_exchange(
      embedded_partitioner, buffer);
    data_exchange.update_ghost_values(src);

    Vector<Number> vector_src, vector_dst;

    // ... 2) loop over patches
    for (unsigned int p = 0; p < patch_matrices.size(); ++p)
      {
        // ... 2a) gather
        const unsigned int dofs_per_cell = patch_indices[p].size();

        vector_src.reinit(dofs_per_cell);
        vector_dst.reinit(dofs_per_cell);

        // TODO: patch indices are *global*
        //       use local indices for faster access like this:
        // for (unsigned int i = 0; i < dofs_per_cell; ++i)
        //   vector_src[i] = src.local_element(patch_indices[p][i]);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          vector_src[i] = src[patch_indices[p][i]];

        // ... 2b) apply preconditioner
        patch_matrices[p].vmult(vector_dst, vector_src);

        // ... 2c) scatter
        // for (unsigned int i = 0; i < dofs_per_cell; ++i)
        //   dst.local_element(patch_indices[p][i]) += vector_dst[i];
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[patch_indices[p][i]] += vector_dst[i];
      }

    // ... 3) compress
    data_exchange.compress(dst);
    data_exchange.zero_out_ghost_values(src);
  }

private:
  const DoFHandler<dim, spacedim> &dof_handler;

  // ASM
  std::vector<std::vector<types::global_dof_index>> patch_indices;
  std::vector<FullMatrix<Number>>                   patch_matrices;

  // inverse diagonal
  VectorType inverse_diagonal;

  const WeightingType weighting_type;

  // embedded partitioner
  std::shared_ptr<const Utilities::MPI::Partitioner> embedded_partitioner;
  mutable AlignedVector<Number>                      buffer;
};

DEAL_II_NAMESPACE_CLOSE

#endif
