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

#ifndef precondition_patch_indices_h
#define precondition_patch_indices_h


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>


template <int dim, int spacedim, typename Number>
std::vector<std::vector<dealii::types::global_dof_index>>
prepare_patch_indices(const dealii::DoFHandler<dim, spacedim> &dof_handler,
                      const dealii::AffineConstraints<Number> &constraints)
{
  std::vector<std::vector<dealii::types::global_dof_index>> patch_indices;

  std::vector<dealii::types::global_dof_index> local_indices;
  for (const auto &cell : dof_handler.active_cell_iterators() | dealii::IteratorFilters::LocallyOwnedCell())
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

                // check faces among locally owned and ghost cells
                // to cover all patches that possibly contain locally active dofs
                if (neighbor_subface->is_locally_owned() || neighbor_subface->is_ghost())
                  // problem criterion: cell faces h-refined cell with lower polynomial degree
                  // (corresponds to 'version 6' in previous experiments)
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

          std::vector<dealii::types::global_dof_index> local_unconstrained_indices;
          for (const auto i : local_indices)
            if (constraints.is_constrained(i) == false)
              local_unconstrained_indices.emplace_back(i);

          if (local_unconstrained_indices.empty() == false)
            patch_indices.push_back(std::move(local_unconstrained_indices))            }
        }

  return patch_indices;
}


#endif
