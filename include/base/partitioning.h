// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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

#ifndef base_partitioning_h
#define base_partitioning_h


#include <deal.II/base/index_set.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <vector>


class Partitioning
{
public:
  Partitioning() = default;

  template <int dim, int spacedim>
  void
  reinit(const dealii::DoFHandler<dim, spacedim> &dof_handler);

  const dealii::IndexSet &
  get_owned_dofs() const;
  const dealii::IndexSet &
  get_relevant_dofs() const;

private:
  dealii::IndexSet owned_dofs;
  dealii::IndexSet relevant_dofs;
};

template <int dim, int spacedim>
void
Partitioning::reinit(const dealii::DoFHandler<dim, spacedim> &dof_handler)
{
  owned_dofs    = dof_handler.locally_owned_dofs();
  relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
}

inline const dealii::IndexSet &
Partitioning::get_owned_dofs() const
{
  return owned_dofs;
}

inline const dealii::IndexSet &
Partitioning::get_relevant_dofs() const
{
  return relevant_dofs;
}



class BlockPartitioning
{
public:
  BlockPartitioning() = default;

  template <int dim, int spacedim>
  void
  reinit(const std::vector<dealii::DoFHandler<dim, spacedim>> dof_handlers)

  const dealii::IndexSet &
  get_owned_dofs() const;
  const dealii::IndexSet &
  get_relevant_dofs() const;

  unsigned int
  get_n_blocks() const;
  const std::vector<Partitioning> &
  get_partitioning_per_block() const;
  const std::vector<IndexSet> &
  get_owned_dofs_per_block() const;
  const std::vector<IndexSet> &
  get_relevant_dofs_per_block() const;

private:
  dealii::IndexSet owned_dofs;
  dealii::IndexSet relevant_dofs;

  unsigned int              n_blocks;
  std::vector<Partitioning> partitioning_per_block;
  std::vector<IndexSet>     owned_dofs_per_block;
  std::vector<IndexSet>     relevant_dofs_per_block;
}

template <int dim, int spacedim>
void
BlockPartitioning::reinit(const std::vector<dealii::DoFHandler<dim, spacedim>> &dof_handlers)
{
  n_blocks = dof_handlers.size();

  partitioning_per_block.resize(n_blocks);
  for (unsigned int b = 0; b < n_blocks; ++b)
    partitioning_per_block[b].reinit(dof_handlers[b]);

  owned_dofs.clear();
  owned_dofs_per_block.clear();
  relevant_dofs.clear();
  relevant_dofs_per_block.clear();
  for (const auto &partitioning : partitioning_per_block)
    {
      owned_dofs.add_indices(partitioning.get_owned_dofs());
      owned_dofs_per_block.push_back(partitioning.get_owned_dofs());

      relevant_dofs.add_indices(partitioning.get_relevant_dofs());
      relevant_dofs_per_block.push_back(partitioning.get_relevant_dofs());
    }
}

inline const dealii::IndexSet &
BlockPartitioning::get_owned_dofs() const
{
  return owned_dofs;
}

inline const dealii::IndexSet &
BlockPartitioning::get_relevant_dofs() const
{
  return relevant_dofs;
}

inline unsigned int
BlockPartitioning::get_n_blocks() const
{
  return n_blocks;
}

inline const std::vector<Partitioning> &
BlockPartitioning::get_partitioning_per_block() const
{
  return partitioning_per_block;
}

inline const std::vector<IndexSet> &
BlockPartitioning::get_owned_dofs_per_block() const
{
  return owned_dofs_per_block;
}

inline const std::vector<IndexSet> &
BlockPartitioning::get_relevant_dofs_per_block() const
{
  return relevant_dofs_per_block;
}


#endif
