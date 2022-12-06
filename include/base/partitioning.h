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


#include <deal.II/base/config.h>

#include <deal.II/base/index_set.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <numeric>
#include <vector>


class Partitioning
{
public:
  Partitioning() = default;

  template <int dim, int spacedim>
  void
  reinit(const dealii::DoFHandler<dim, spacedim> &dof_handler,
         const std::vector<unsigned int>         &target_block = {});

  const dealii::IndexSet &
  get_owned_dofs() const;
  const dealii::IndexSet &
  get_relevant_dofs() const;

  unsigned int
  get_n_blocks() const;
  const std::vector<dealii::IndexSet> &
  get_owned_dofs_per_block() const;
  const std::vector<dealii::IndexSet> &
  get_relevant_dofs_per_block() const;

private:
  dealii::IndexSet owned_dofs;
  dealii::IndexSet relevant_dofs;

  unsigned int                  n_blocks;
  std::vector<dealii::IndexSet> owned_dofs_per_block;
  std::vector<dealii::IndexSet> relevant_dofs_per_block;
};

template <int dim, int spacedim>
void
Partitioning::reinit(const dealii::DoFHandler<dim, spacedim> &dof_handler,
                     const std::vector<unsigned int>         &target_block)
{
  owned_dofs    = dof_handler.locally_owned_dofs();
  relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);

  if (target_block.size() > 0)
    {
      const std::vector<dealii::types::global_dof_index> dofs_per_block =
        dealii::DoFTools::count_dofs_per_fe_block(dof_handler, target_block);

      n_blocks = dofs_per_block.size();

      std::vector<dealii::types::global_dof_index> cumulated_dofs_per_block(n_blocks + 1, 0);
      std::partial_sum(dofs_per_block.begin(),
                       dofs_per_block.end(),
                       std::next(cumulated_dofs_per_block.begin()));

      owned_dofs_per_block.resize(n_blocks);
      relevant_dofs_per_block.resize(n_blocks);
      for (unsigned int b = 0; b < n_blocks; ++b)
        {
          const auto &block_start = cumulated_dofs_per_block[b];
          const auto &block_end   = cumulated_dofs_per_block[b + 1];

          owned_dofs_per_block[b]    = this->owned_dofs.get_view(block_start, block_end);
          relevant_dofs_per_block[b] = this->relevant_dofs.get_view(block_start, block_end);
        }
    }
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

inline unsigned int
Partitioning::get_n_blocks() const
{
  return n_blocks;
}

inline const std::vector<dealii::IndexSet> &
Partitioning::get_owned_dofs_per_block() const
{
  Assert(n_blocks > 0, dealii::ExcMessage("No blocks initialized."));

  return owned_dofs_per_block;
}

inline const std::vector<dealii::IndexSet> &
Partitioning::get_relevant_dofs_per_block() const
{
  Assert(n_blocks > 0, dealii::ExcMessage("No blocks initialized."));

  return relevant_dofs_per_block;
}


#endif
