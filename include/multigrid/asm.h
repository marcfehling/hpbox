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

#ifndef multigrid_asm_h
#define multigrid_asm_h


#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/sparse_matrix_tools.h>

#include <global.h>


// NOTE:
// Adopted from
// https://github.com/peterrum/dealii-asm/blob/d998b9b344a19c9d2890e087f953c2f93e6546ae/include/preconditioners.h#L145.


DEAL_II_NAMESPACE_OPEN

template <typename VectorType>
class PreconditionASM
{
private:
  enum class WeightingType
  {
    none,
    left,
    right,
    symm
  };

  const WeightingType weighting_type = WeightingType::symm;

  using Number = typename VectorType::value_type;

public:
  PreconditionASM(const std::vector<std::vector<types::global_dof_index>> &patch_indices)
    : indices(patch_indices)
  {}

  PreconditionASM(std::vector<std::vector<types::global_dof_index>> &&patch_indices)
    : indices(std::move(patch_indices))
  {}

  template <typename GlobalSparseMatrixType, typename GlobalSparsityPattern>
  void
  initialize(const GlobalSparseMatrixType &global_sparse_matrix,
             const GlobalSparsityPattern  &global_sparsity_pattern,
             const Partitioning           &partitioning)
  {
    TimerOutput::Scope t(getTimer(), "initialize_asm");

    // treat unprocessed DoFs as blocks of size 1x1

    // TODO: Move this part into the constructor

    // ATTENTION: This function modifies indices. Do not call this twice!

    VectorType unprocessed_indices(partitioning.get_owned_dofs(),
                                   partitioning.get_relevant_dofs(),
                                   partitioning.get_communicator());

    // 'indices' contains global indices on locally owned cells
    for (const auto &indices_i : indices)
      for (const auto i : indices_i)
        unprocessed_indices[i]++;

    unprocessed_indices.compress(VectorOperation::add);

    for (const auto &i : unprocessed_indices.locally_owned_elements())
      if (unprocessed_indices[i] == 0)
        indices.emplace_back(std::vector<types::global_dof_index>{i});

    //
    // build blocks
    //
    SparseMatrixTools::restrict_to_full_matrices(global_sparse_matrix,
                                                 global_sparsity_pattern,
                                                 indices,
                                                 blocks);

    for (auto &block : blocks)
      if (block.m() > 0 && block.n() > 0)
        block.gauss_jordan();

    //
    // prepare weights
    //
    VectorType     weights;
    Vector<Number> vector_weights;
    if (weighting_type != WeightingType::none)
      {
        weights.reinit(partitioning.get_owned_dofs(),
                       partitioning.get_relevant_dofs(),
                       partitioning.get_communicator());

        for (unsigned int c = 0; c < indices.size(); ++c)
          {
            const unsigned int dofs_per_cell = indices[c].size();
            vector_weights.reinit(dofs_per_cell);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              vector_weights[i] = 1.0;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              weights[indices[c][i]] += vector_weights[i];
          }

        weights.compress(VectorOperation::add);
        for (auto &i : weights)
          i = (weighting_type == WeightingType::symm) ? std::sqrt(1.0 / i) : (1.0 / i);
        weights.update_ghost_values();
      }

    //
    // multiply weights to patch matrices (blocks)
    //
    if (weighting_type != WeightingType::none)
      {
        for (unsigned int cell = 0; cell < indices.size(); ++cell)
          {
            const unsigned int dofs_per_cell = indices[cell].size();

            vector_weights.reinit(dofs_per_cell);
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              vector_weights[i] += weights[indices[cell][i]];

            auto &block = blocks[cell];

            if (weighting_type == WeightingType::symm || weighting_type == WeightingType::right)
              {
                // multiply weights from right B(wI), i.e.,
                // multiply one weight for each column
                for (unsigned int r = 0; r < dofs_per_cell; ++r)
                  for (unsigned int c = 0; c < dofs_per_cell; ++c)
                    block(r, c) *= vector_weights[c];
              }

            if (weighting_type == WeightingType::symm || weighting_type == WeightingType::left)
              {
                // multiply weights from left (wI)B, i.e.,
                // multiply one weight for each row
                for (unsigned int r = 0; r < dofs_per_cell; ++r)
                  for (unsigned int c = 0; c < dofs_per_cell; ++c)
                    block(r, c) *= vector_weights[r];
              }
          }
      }
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    TimerOutput::Scope t(getTimer(), "vmult_asm");

    dst = 0.0;
    src.update_ghost_values();

    Vector<Number> vector_src, vector_dst;

    for (unsigned int c = 0; c < indices.size(); ++c)
      {
        const unsigned int dofs_per_cell = indices[c].size();

        vector_src.reinit(dofs_per_cell);
        vector_dst.reinit(dofs_per_cell);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          vector_src[i] += src[indices[c][i]];

        blocks[c].vmult(vector_dst, vector_src);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          dst[indices[c][i]] += vector_dst[i];
      }

    src.zero_out_ghost_values();
    dst.compress(VectorOperation::add);
  }

private:
  // make indices const!
  std::vector<std::vector<types::global_dof_index>> indices;
  std::vector<FullMatrix<Number>>                   blocks;
};

DEAL_II_NAMESPACE_CLOSE


#endif
