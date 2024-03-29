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

#ifndef multigrid_patch_indices_h
#define multigrid_patch_indices_h


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/affine_constraints.h>


template <int dim, int spacedim, typename Number>
std::vector<std::vector<dealii::types::global_dof_index>>
prepare_patch_indices(const dealii::DoFHandler<dim, spacedim> &dof_handler,
                      const dealii::AffineConstraints<Number> &constraints)
{
  std::vector<std::vector<dealii::types::global_dof_index>> patch_indices;

  std::vector<dealii::types::global_dof_index> local_indices;
  for (const auto &cell :
       dof_handler.active_cell_iterators() | dealii::IteratorFilters::LocallyOwnedCell())
    for (const auto f : cell->face_indices())
      if (cell->at_boundary(f) == false)
        {
          bool flag = false;

          if (cell->face(f)->has_children())
            for (unsigned int sf = 0; sf < cell->face(f)->n_children(); ++sf)
              {
                const auto neighbor_subface = cell->neighbor_child_on_subface(f, sf);

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
          cell->face(f)->get_dof_indices(local_indices, cell->active_fe_index());

          for (unsigned int c = 0; c < cell->get_fe().n_components(); ++c)
            {
              std::vector<dealii::types::global_dof_index> local_unconstrained_indices;
              for (unsigned int i = 0; i < local_indices.size(); ++i)
                if (cell->get_fe().face_system_to_component_index(i).first == c)
                  if (constraints.is_constrained(local_indices[i]) == false)
                    local_unconstrained_indices.emplace_back(local_indices[i]);

              if (local_unconstrained_indices.empty() == false)
                patch_indices.push_back(std::move(local_unconstrained_indices));
            }
        }

  return patch_indices;
}


#endif
