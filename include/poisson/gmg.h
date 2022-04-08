// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2022 by the deal.II authors
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

#ifndef poisson_gmg_h
#define poisson_gmg_h


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/mapping_collection.h>

#include <deal.II/lac/solver_control.h>

#include <deal.II/numerics/vector_tools.h>

#include <mg_solver.h>


namespace Poisson
{
  template <typename VectorType,
            typename OperatorType,
            int dim,
            int spacedim,
            typename... Args>
  void
  solve_gmg(
    dealii::SolverControl                             &solver_control,
    const OperatorType                                &system_matrix,
    VectorType                                        &dst,
    const VectorType                                  &src,
    const dealii::hp::MappingCollection<dim, spacedim> mapping_collection,
    const dealii::DoFHandler<dim, spacedim>           &dof_handler,
    Args &&...operator_args)
  {
    using namespace dealii;

    const MGSolverParameters mg_data;

    // Create a DoFHandler and operator for each multigrid level defined
    // by p-coarsening, as well as, create transfer operators.
    MGLevelObject<DoFHandler<dim, spacedim>>           dof_handlers;
    MGLevelObject<std::unique_ptr<OperatorType>>       operators;
    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;

    std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>>
      coarse_grid_triangulations;
    if (mg_data.transfer.perform_h_transfer)
      coarse_grid_triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          dof_handler.get_triangulation());
    else
      coarse_grid_triangulations.emplace_back(
        const_cast<Triangulation<dim, spacedim> *>(
          &(dof_handler.get_triangulation())),
        [](auto &) {
          // empty deleter, since fine_triangulation_in is an external field
          // and its destructor is called somewhere else
        });

    const unsigned int n_h_levels = coarse_grid_triangulations.size() - 1;

    // Determine the number of levels.
    const auto get_max_active_fe_degree = [&](const auto &dof_handler) {
      unsigned int max = 0;

      for (auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          max =
            std::max(max, dof_handler.get_fe(cell->active_fe_index()).degree);

      return Utilities::MPI::max(max, MPI_COMM_WORLD);
    };

    const unsigned int n_p_levels =
      MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
        get_max_active_fe_degree(dof_handler), mg_data.transfer.p_sequence)
        .size();

    std::map<unsigned int, unsigned int> fe_index_for_degree;
    for (unsigned int i = 0; i < dof_handler.get_fe_collection().size(); ++i)
      {
        const unsigned int degree = dof_handler.get_fe(i).degree;
        Assert(fe_index_for_degree.find(degree) == fe_index_for_degree.end(),
               ExcMessage("FECollection does not contain unique degrees."));
        fe_index_for_degree[degree] = i;
      }

    unsigned int minlevel   = 0;
    unsigned int minlevel_p = n_h_levels;
    unsigned int maxlevel   = n_h_levels + n_p_levels - 1;

    // Allocate memory for all levels.
    dof_handlers.resize(minlevel, maxlevel);
    operators.resize(minlevel, maxlevel);
    transfers.resize(minlevel, maxlevel);

    // Loop from max to min level and set up DoFHandler with coarser mesh...
    for (unsigned int l = 0; l < n_h_levels; ++l)
      {
        dof_handlers[l].reinit(*coarse_grid_triangulations[l]);
        dof_handlers[l].distribute_dofs(dof_handler.get_fe_collection());
      }

    // ... with lower polynomial degrees
    for (unsigned int i = 0, l = maxlevel; i < n_p_levels; ++i, --l)
      {
        dof_handlers[l].reinit(dof_handler.get_triangulation());

        if (l == maxlevel) // finest level
          {
            auto &dof_handler_mg = dof_handlers[l];

            auto cell_other = dof_handler.begin_active();
            for (auto &cell : dof_handler_mg.active_cell_iterators())
              {
                if (cell->is_locally_owned())
                  cell->set_active_fe_index(cell_other->active_fe_index());
                cell_other++;
              }
          }
        else // coarse level
          {
            auto &dof_handler_fine   = dof_handlers[l + 1];
            auto &dof_handler_coarse = dof_handlers[l + 0];

            auto cell_other = dof_handler_fine.begin_active();
            for (auto &cell : dof_handler_coarse.active_cell_iterators())
              {
                if (cell->is_locally_owned())
                  {
                    const unsigned int next_degree =
                      MGTransferGlobalCoarseningTools::
                        create_next_polynomial_coarsening_degree(
                          cell_other->get_fe().degree,
                          mg_data.transfer.p_sequence);
                    Assert(fe_index_for_degree.find(next_degree) !=
                             fe_index_for_degree.end(),
                           ExcMessage("Next polynomial degree in sequence "
                                      "does not exist in FECollection."));

                    cell->set_active_fe_index(fe_index_for_degree[next_degree]);
                  }
                cell_other++;
              }
          }

        dof_handlers[l].distribute_dofs(dof_handler.get_fe_collection());
      }

    // Create data structures on each multigrid level.
    MGLevelObject<AffineConstraints<typename VectorType::value_type>>
      constraints(minlevel, maxlevel);

    for (unsigned int level = minlevel; level <= maxlevel; level++)
      {
        const auto &dof_handler = dof_handlers[level];
        auto       &constraint  = constraints[level];

        // ... constraints (with homogenous Dirichlet BC)
        {
          IndexSet locally_relevant_dofs;
          DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                  locally_relevant_dofs);
          constraint.reinit(locally_relevant_dofs);


          DoFTools::make_hanging_node_constraints(dof_handler, constraint);
          VectorTools::interpolate_boundary_values(
            mapping_collection,
            dof_handler,
            0,
            Functions::ZeroFunction<dim>(),
            constraint);
          constraint.close();
        }

        // ... operator (just like on the finest level)
        {
          VectorType dummy;

          operators[level] = std::make_unique<OperatorType>(
            std::forward<Args>(operator_args)...);

          operators[level]->reinit(dof_handler, constraint, dummy);
        }
      }

    // Set up intergrid operators.
    for (unsigned int level = minlevel; level < minlevel_p; ++level)
      transfers[level + 1].reinit_geometric_transfer(dof_handlers[level + 1],
                                                     dof_handlers[level],
                                                     constraints[level + 1],
                                                     constraints[level]);

    for (unsigned int level = minlevel_p; level < maxlevel; ++level)
      transfers[level + 1].reinit_polynomial_transfer(dof_handlers[level + 1],
                                                      dof_handlers[level],
                                                      constraints[level + 1],
                                                      constraints[level]);

    // Collect transfer operators within a single operator as needed by
    // the Multigrid solver class.
    MGTransferGlobalCoarsening<dim, VectorType> transfer(
      transfers, [&](const auto l, auto &vec) {
        operators[l]->initialize_dof_vector(vec);
      });

    // Proceed to solve the problem with multigrid.
    mg_solve(solver_control,
             dst,
             src,
             mg_data,
             dof_handler,
             system_matrix,
             operators,
             transfer);
  }
} // namespace Poisson


#endif
