// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2023 by the deal.II authors
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


#include <deal.II/base/mpi.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <global.h>
#include <linear_algebra.h>
#include <log.h>

#include <algorithm>
#include <vector>

using namespace dealii;


namespace Log
{
  void
  log_cycle(const unsigned int cycle, const Parameter &prm)
  {
    TableHandler &table = getTable();

    getPCOut() << "Cycle " << cycle << ':' << std::endl;
    table.add_value("cycle", cycle);

    table.add_value("processes", Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
    table.add_value("stem", prm.file_stem);
    table.add_value("weighting_exponent", prm.prm_adaptation.weighting_exponent);
  }



  template <int dim, typename T, int spacedim>
  void
  log_hp_diagnostics(const parallel::distributed::Triangulation<dim, spacedim> &triangulation,
                     const DoFHandler<dim, spacedim>                           &dof_handler,
                     const AffineConstraints<T>                                &constraint)
  {
    log_hp_diagnostics<dim, T, spacedim>(triangulation, {&dof_handler}, {&constraint});
  }



  template <int dim, typename T, int spacedim>
  void
  log_hp_diagnostics(const parallel::distributed::Triangulation<dim, spacedim> &triangulation,
                     const std::vector<const DoFHandler<dim, spacedim> *>      &dof_handlers,
                     const std::vector<const AffineConstraints<T> *>           &constraints)
  {
    Assert((!dof_handlers.empty()) && (!constraints.empty()), ExcMessage("Empty containers"));

    ConditionalOStream &pcout = getPCOut();
    TableHandler       &table = getTable();

    const MPI_Comm &mpi_communicator = triangulation.get_communicator();

    const unsigned int first_n_processes =
      std::min<unsigned int>(8, Utilities::MPI::n_mpi_processes(mpi_communicator));
    const bool output_cropped =
      first_n_processes < Utilities::MPI::n_mpi_processes(mpi_communicator);

    const auto pcout_first_n = [&pcout, first_n_processes, output_cropped](const auto &container) {
      for (unsigned int i = 0; i < first_n_processes; ++i)
        pcout << ' ' << container[i];
      if (output_cropped)
        pcout << " ...";
      pcout << std::endl;
    };

    {
      pcout << "   Number of global levels:      " << triangulation.n_global_levels() << std::endl;
      table.add_value("global_levels", triangulation.n_global_levels());
    }

    {
      pcout << "   Number of active cells:       " << triangulation.n_global_active_cells()
            << std::endl;
      table.add_value("active_cells", triangulation.n_global_active_cells());

      const std::vector<unsigned int> n_active_cells_per_subdomain =
        Utilities::MPI::gather(mpi_communicator, triangulation.n_locally_owned_active_cells());

      pcout << "     by partition:              ";
      pcout_first_n(n_active_cells_per_subdomain);
    }

    {
      // log active fe indices only for the first dof_handler
      const DoFHandler<dim, spacedim>       &dof_handler   = *dof_handlers[0];
      const hp::FECollection<dim, spacedim> &fe_collection = dof_handler.get_fe_collection();

      std::vector<types::global_cell_index> n_fe_indices(fe_collection.size(), 0);
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          ++n_fe_indices[cell->active_fe_index()];

      Utilities::MPI::sum(n_fe_indices, mpi_communicator, n_fe_indices);

      const auto max_nonzero = std::find_if(std::crbegin(n_fe_indices),
                                            std::crend(n_fe_indices),
                                            [](const auto &i) { return i > 0; });
      Assert(max_nonzero != std::crend(n_fe_indices), ExcInternalError());
      const types::fe_index max_nonzero_fe_index =
        std::distance(n_fe_indices.cbegin(), (max_nonzero + 1).base());

      pcout << "   Max active FE index in use  : " << max_nonzero_fe_index << std::endl;
      table.add_value("max_fe_index", max_nonzero_fe_index);

      pcout << "   Frequencies of poly. degrees:";
      for (unsigned int i = 0; i < fe_collection.size(); ++i)
        if (n_fe_indices[i] > 0)
          pcout << ' ' << fe_collection[i].degree << ":" << n_fe_indices[i];
      pcout << std::endl;
    }

    types::global_dof_index global_dofs = 0;
    {
      for (const auto &dof_handler : dof_handlers)
        global_dofs += dof_handler->n_dofs();

      types::global_dof_index n_locally_owned_dofs = 0;
      for (const auto &dof_handler : dof_handlers)
        n_locally_owned_dofs += dof_handler->n_locally_owned_dofs();
      const std::vector<types::global_dof_index> n_dofs_per_subdomain =
        Utilities::MPI::gather(mpi_communicator, n_locally_owned_dofs);

      pcout << "   Number of degrees of freedom: " << global_dofs << std::endl;
      table.add_value("dofs", global_dofs);

      pcout << "     by partition:              ";
      pcout_first_n(n_dofs_per_subdomain);
    }

    {
      types::global_dof_index n_constraints = 0;
      for (const auto constraint : constraints)
        n_constraints += constraint->n_constraints();

      const std::vector<types::global_dof_index> n_constraints_per_subdomain =
        Utilities::MPI::gather(mpi_communicator, n_constraints);
      const types::global_dof_index global_constraints =
        std::accumulate(n_constraints_per_subdomain.begin(), n_constraints_per_subdomain.end(), 0);

      pcout << "   Number of constraints:        " << global_constraints << std::endl;
      table.add_value("constraints", global_constraints);

      pcout << "     by partition:              ";
      pcout_first_n(n_constraints_per_subdomain);

      const float fraction = static_cast<float>(global_constraints) / global_dofs;
      pcout << "   Fraction of constraints:      " << 100 * fraction << "%" << std::endl;
    }

    {
      types::global_dof_index n_identities = 0;
      for (const auto constraint : constraints)
        n_identities += constraint->n_identities();

      const std::vector<types::global_dof_index> n_identities_per_subdomain =
        Utilities::MPI::gather(mpi_communicator, n_identities);
      const types::global_dof_index global_identities =
        std::accumulate(n_identities_per_subdomain.begin(), n_identities_per_subdomain.end(), 0);

      pcout << "   Number of identities:         " << global_identities << std::endl;
      table.add_value("identities", global_identities);

      pcout << "     by partition:              ";
      pcout_first_n(n_identities_per_subdomain);

      const float fraction = static_cast<float>(global_identities) / global_dofs;
      pcout << "   Fraction of identities:       " << 100 * fraction << "%" << std::endl;
    }
  }



  void
  log_iterations(const SolverControl &control)
  {
    getPCOut() << "   Solved in " << control.last_step() << " iterations." << std::endl;
    getTable().add_value("iterations", control.last_step());
  }



  template <typename MatrixType>
  void
  log_nonzero_elements(const MatrixType &matrix)
  {
    getPCOut() << "   Number of nonzero elements:   " << matrix.n_nonzero_elements() << std::endl;
    getTable().add_value("nonzero_elements", matrix.n_nonzero_elements());
  }



  void
  log_timing_statistics(const MPI_Comm mpi_communicator)
  {
    getTimer().print_wall_time_statistics(mpi_communicator);
    getPCOut() << std::endl;

    for (const auto &summary : getTimer().get_summary_data(TimerOutput::total_wall_time))
      {
        const Utilities::MPI::MinMaxAvg data =
          Utilities::MPI::min_max_avg(summary.second, mpi_communicator);

        getTable().add_value(summary.first + "_min", data.min);
        getTable().add_value(summary.first + "_max", data.max);
        getTable().add_value(summary.first + "_avg", data.avg);
        getTable().set_scientific(summary.first + "_min", true);
        getTable().set_scientific(summary.first + "_max", true);
        getTable().set_scientific(summary.first + "_avg", true);
      }
  }



  template <int dim, int spacedim>
  void
  log_patch_dofs(const std::vector<std::vector<types::global_dof_index>> &patch_indices,
                 const DoFHandler<dim, spacedim>                         &dof_handler)
  {
    std::set<types::global_dof_index> all_indices;

    // patch_indices contains locally active dofs. patches are unique among all processes, but not
    // all patch indices. so ideally we would need to exchange all patch indices via MPI. currently,
    // it is just an estimate.
    for (const auto &indices : patch_indices)
      for (const auto i : indices)
        all_indices.insert(i);

    const auto n_global_patch_dofs =
      Utilities::MPI::sum<types::global_dof_index>(all_indices.size(),
                                                   dof_handler.get_communicator());

    getPCOut() << "   Number of patch DoFs:         " << n_global_patch_dofs << std::endl;
    getTable().add_value("patch_dofs", n_global_patch_dofs);

    const float fraction = static_cast<float>(n_global_patch_dofs) / dof_handler.n_dofs();
    getPCOut() << "   Fraction of patch DoFs:       " << 100 * fraction << "%" << std::endl;
  }



  // explicit instantiations
  template void
  log_hp_diagnostics<2, double, 2>(const parallel::distributed::Triangulation<2, 2> &,
                                   const DoFHandler<2, 2> &,
                                   const AffineConstraints<double> &);
  template void
  log_hp_diagnostics<3, double, 3>(const parallel::distributed::Triangulation<3, 3> &,
                                   const DoFHandler<3, 3> &,
                                   const AffineConstraints<double> &);

  template void
  log_patch_dofs<2, 2>(const std::vector<std::vector<dealii::types::global_dof_index>> &,
                       const DoFHandler<2, 2> &);
  template void
  log_patch_dofs<3, 3>(const std::vector<std::vector<dealii::types::global_dof_index>> &,
                       const DoFHandler<3, 3> &);

#ifdef DEAL_II_WITH_TRILINOS
  template void
  log_nonzero_elements<TrilinosWrappers::SparseMatrix>(const TrilinosWrappers::SparseMatrix &);
  template void
  log_nonzero_elements<TrilinosWrappers::BlockSparseMatrix>(
    const TrilinosWrappers::BlockSparseMatrix &);
#endif

#ifdef DEAL_II_WITH_PETSC
  template void
  log_nonzero_elements<PETScWrappers::MPI::SparseMatrix>(const PETScWrappers::MPI::SparseMatrix &);
  template void
  log_nonzero_elements<PETScWrappers::MPI::BlockSparseMatrix>(
    const PETScWrappers::MPI::BlockSparseMatrix &);
#endif

} // namespace Log
