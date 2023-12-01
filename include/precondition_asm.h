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


/*
 * NOTE:
 * Adopted from
 * https://github.com/peterrum/dealii-asm/blob/d998b9b344a19c9d2890e087f953c2f93e6546ae/include/preconditioners.h#L145.
 */

#ifndef precondition_asm_h
#define precondition_asm_h


#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix_tools.h>

#include <global.h>

#include <vector>

DEAL_II_NAMESPACE_OPEN


template <typename VectorType, int dim, int spacedim>
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

public:
  PreconditionASM(const DoFHandler<dim, spacedim> &dof_handler,
                  const AffineConstraints<double> &affine_constraints)
    : dof_handler(dof_handler)
    , affine_constraints(affine_constraints)
    , weighting_type(WeightingType::symm)
  {}

  template <typename GlobalSparseMatrixType, typename GlobalSparsityPattern>
  void
  initialize(const GlobalSparseMatrixType &global_sparse_matrix,
             const GlobalSparsityPattern  &global_sparsity_pattern)
  {
    TimerOutput::Scope t(getTimer(), "initialize_asm");

    // patch types
    //   (0) -> cell-centric patches on cells                       X
    //
    //   (1) -> cell-centric patches on cells at refinement levels  X
    //   (2) -> cell-centric patches on cells with coarser neighbor
    //   (3) -> cell-centric patches on cells with finer neighbor   X
    //
    //   (4) -> face-centric patches on cells at refinement levels  X
    //   (5) -> face-centric patches on cells with coarser neighbor
    //   (6) -> face-centric patches on cells with finer neighbor   X
    //
    //   (7) -> edge-centric patches on cells at refinement levels
    //   (8) -> edge-centric patches on cells with coarser neighbor
    //   (9) -> edge-centric patches on cells with finer neighbor   o
    //
    // DoFs not assigned to a patch are implicitly treated as blocks
    // of size 1x1.

    const unsigned int version = 6;

    const auto push_back =
      [&](const std::vector<types::global_dof_index> &indices_local) {
        std::vector<types::global_dof_index> temp;

        for (const auto i : indices_local)
          if (affine_constraints.is_constrained(i) == false)
            temp.emplace_back(i);

        if (temp.empty() == false)
          indices.push_back(temp);
      };

    if (version == 0)
      {
        std::vector<types::global_dof_index> indices_local;

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned() == false)
              continue;

            indices_local.resize(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(indices_local);
            push_back(indices_local);
          }
      }
    else if (version == 1 || version == 2 || version == 3)
      {
        std::vector<types::global_dof_index> indices_local;

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned() == false)
              continue;

            bool flag = false;

            for (const auto f : cell->face_indices())
              if (cell->at_boundary(f) == false)
                {
                  if ((version == 1 || version == 2) &&
                      (cell->level() > cell->neighbor(f)->level()))
                    flag = true;

                  if ((version == 1 || version == 3) &&
                      (cell->neighbor(f)->has_children()))
                    flag = true;
                }

            if (flag == false)
              continue;

            indices_local.resize(cell->get_fe().n_dofs_per_cell());
            cell->get_dof_indices(indices_local);
            push_back(indices_local);
          }
      }
    else if (version == 4 || version == 5 || version == 6)
      {
        std::vector<types::global_dof_index> indices_local;

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned() == false)
              continue;

            for (const auto f : cell->face_indices())
              if (cell->at_boundary(f) == false)
                {
                  bool flag = false;

                  if ((version == 4 || version == 5) &&
                      (cell->level() > cell->neighbor(f)->level()))
                    flag = true;

                  if ((version == 4 || version == 6) &&
                      (cell->face(f)->has_children()))
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
                  push_back(indices_local);
                }
          }
      }
    else if ((dim == 3) && (version == 7 || version == 8 || version == 9))
      {
        std::vector<types::global_dof_index> indices_local;
        std::set<unsigned int>               processed_lines;

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned() == false)
              continue;

            for (const auto f : cell->face_indices())
              if (cell->at_boundary(f) == false)
                {
                  bool flag = false;

                  if ((version == 7 || version == 8) &&
                      (cell->level() > cell->neighbor(f)->level()))
                    flag = true;

                  if ((version == 7 || version == 9) &&
                      (cell->neighbor(f)->has_children()))
                    flag = true;

                  if (flag == false)
                    continue;

                  for (const auto l : cell->face(f)->line_indices())
                    {
                      const bool contains = processed_lines.find(cell->face(f)->line(l)->index()) != processed_lines.end();
                      if (contains)
                        continue;

                      indices_local.resize(
                        cell->get_fe().n_dofs_per_line() +
                        2 * cell->get_fe().n_dofs_per_vertex());

                      cell->face(f)->line(l)->get_dof_indices(
                        indices_local, cell->active_fe_index());

                      push_back(indices_local);

                      processed_lines.insert(cell->face(f)->line(l)->index());
                    }
                }
          }
      }

    // treat unprocessed DoFs as blocks of size 1x1
    const IndexSet relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
    VectorType unprocessed_indices(dof_handler.locally_owned_dofs(), relevant, dof_handler.get_communicator());

    // 'indices' only contains locally owned indices
    for (const auto &indices_i : indices)
      for (const auto i : indices_i)
        unprocessed_indices[i]++;

    unprocessed_indices.compress(VectorOperation::add);

    for (const auto &i : unprocessed_indices.locally_owned_elements())
      if (unprocessed_indices[i] == 0)
        indices.emplace_back(std::vector<types::global_dof_index>{i});

    if (false)
      {
        for (const auto &indices_i : indices)
          {
            for (const auto i : indices_i)
              std::cout << i << " ";
            std::cout << std::endl;
          }
      }

    //
    // build blocks
    //
    SparseMatrixTools::restrict_to_full_matrices(global_sparse_matrix,
                                                 global_sparsity_pattern,
                                                 indices,
                                                 blocks);

    for (auto &block : blocks)
      {
        if (false && (block.m() > 1))
          {
            block.print_formatted(std::cout, 2, false, 5, "0.00");
            std::cout << std::endl;
          }
        if (block.m() > 0 && block.n() > 0)
          block.gauss_jordan();
      }

    //
    // prepare weights
    //
    Vector<double> vector_weights;
    if (weighting_type != WeightingType::none)
      {
        const IndexSet relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
        weights.reinit(dof_handler.locally_owned_dofs(), relevant, dof_handler.get_communicator());

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
          i = (weighting_type == WeightingType::symm) ? std::sqrt(1.0 / i) :
                                                        (1.0 / i);
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

            auto & block = blocks[cell];

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
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    TimerOutput::Scope t(getTimer(), "vmult_asm");

    dst = 0.0;
    src.update_ghost_values();

    Vector<double> vector_src, vector_dst;

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
  const DoFHandler<dim, spacedim> &dof_handler;
  const AffineConstraints<double> &affine_constraints;

  std::vector<std::vector<types::global_dof_index>>        indices;
  std::vector<FullMatrix<typename VectorType::value_type>> blocks;

  const WeightingType weighting_type;
  VectorType          weights;
};

DEAL_II_NAMESPACE_CLOSE

#endif
