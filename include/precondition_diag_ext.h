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

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/sparse_matrix_tools.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.templates.h>

#include <global.h>

DEAL_II_NAMESPACE_OPEN


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
  ExtendedDiagonalPreconditioner(const DoFHandler<dim, spacedim> &dof_handler,
                                 const AffineConstraints<double> &affine_constraints)
    : dof_handler(dof_handler)
    , affine_constraints(affine_constraints)
    , weighting_type(WeightingType::symm)
  {}

  template <typename GlobalSparseMatrixType, typename GlobalSparsityPattern>
  void
  initialize(const GlobalSparseMatrixType &global_sparse_matrix,
             const GlobalSparsityPattern  &global_sparsity_pattern,
             const VectorType             &inverse_diagonal)
  {
    TimerOutput::Scope t(getTimer(), "initialize_extdiag");

    const auto large_partitioner = inverse_diagonal.get_partitioner();
    // TODO: this should only work for LA distributed Vector,
    //       add a corresponding function to other vector types?

    // only use "version 6" to identify patch indices
    std::vector<types::global_dof_index> indices_local;
    for (const auto &cell : dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
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

                  if(neighbor_subface->get_fe().degree < cell->get_fe().degree)
                    flag = true;
                }

            if (flag == false)
              continue;

            indices_local.resize(cell->get_fe().n_dofs_per_face());
            cell->face(f)->get_dof_indices(indices_local,
                                           cell->active_fe_index());
            // indices_local now contains *global* indices

            std::vector<types::global_dof_index> temp;
            for (const auto i : indices_local)
              if (affine_constraints.is_constrained(i) == false)
                temp.emplace_back(i);

            if (temp.empty() == false)
              patch_indices.push_back(temp);
          }

    //
    // build patch matrices
    //
    // TODO: partial assembly on provided patch_indices
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
    this->inverse_diagonal = inverse_diagonal;

    std::vector<types::global_dof_index> ghost_indices;

    // 'patch_indices' contains global indices on locally owned cells
    for (const auto &indices : patch_indices)
      for (const auto i : indices)
        {
          if (large_partitioner->in_local_range(i))
            this->inverse_diagonal[i] = 0.0;
          else
            ghost_indices.push_back(i);
        }

    //
    // set embedded partitioner
    //
//    std::sort(ghost_indices.begin(), ghost_indices.end());
//    ghost_indices.erase(std::unique(ghost_indices.begin(), ghost_indices.end()),
//                        ghost_indices.end());
//
//    IndexSet ghost_indices_is(large_partitioner->size());
//    ghost_indices_is.add_indices(ghost_indices.begin(), ghost_indices.end());
//
//    const auto partitioner =
//      std::make_shared<const Utilities::MPI::Partitioner>(
//        large_partitioner->locally_owned_range(),
//        ghost_indices_is,
//        large_partitioner->get_mpi_communicator());
//
//    this->embedded_partitioner =
//      internal::create_embedded_partitioner(partitioner, large_partitioner);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    TimerOutput::Scope t(getTimer(), "vmult_extdiag");

    // apply inverse diagonal
    internal::DiagonalMatrix::assign_and_scale(dst, src, this->inverse_diagonal);

    // apply ASM: 1) update ghost values
    src.update_ghost_values();
//    internal::SimpleVectorDataExchange<Number> data_exchange(
//      embedded_partitioner, buffer);
//    data_exchange.update_ghost_values(src);

    Vector<Number> vector_src, vector_dst;

    // ... 2) loop over patches
    for (unsigned int p = 0; p < patch_matrices.size(); ++p)
      {
        // ... 2a) gather
        const unsigned int dofs_per_cell = patch_indices[p].size();

        vector_src.reinit(dofs_per_cell);
        vector_dst.reinit(dofs_per_cell);

        // TODO: patch indices are *global*
        //       use local indices for faster access (see below)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          vector_src[i] = src[patch_indices[p][i]];

        // ... 2b) apply preconditioner
        patch_matrices[p].vmult(vector_dst, vector_src);

        // ... 2c) scatter
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[patch_indices[p][i]] += vector_dst[i];

//        for (unsigned int i = 0; i < dofs_per_cell; ++i)
//          vector_src[i] = src.local_element(patch_indices[p][i]);
//
//        // ... 2b) apply preconditioner
//        patch_matrices[p].vmult(vector_dst, vector_src);
//
//        // ... 2c) scatter
//        for (unsigned int i = 0; i < dofs_per_cell; ++i)
//          dst.local_element(patch_indices[p][i]) += vector_dst[i];
      }

    // ... 3) compress
    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();
//    data_exchange.compress(dst);
//    data_exchange.zero_out_ghost_values(src);
  }

private:
  const DoFHandler<dim, spacedim> &dof_handler;
  const AffineConstraints<double> &affine_constraints;

  // inverse diagonal
  VectorType inverse_diagonal;

  // ASM
  std::vector<std::vector<types::global_dof_index>> patch_indices;
  std::vector<FullMatrix<Number>>                   patch_matrices;

  const WeightingType weighting_type;

  // embedded partitioner
//  std::shared_ptr<const Utilities::MPI::Partitioner> embedded_partitioner;
//  mutable AlignedVector<Number>                      buffer;
};

DEAL_II_NAMESPACE_CLOSE

#endif
