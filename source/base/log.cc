// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2022 by the deal.II authors
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

#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <base/global.h>
#include <base/linear_algebra.h>
#include <base/log.h>

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
  log_timings()
  {
    getTimer().print_summary();
    getPCOut() << std::endl;

    for (const auto &summary : getTimer().get_summary_data(TimerOutput::total_wall_time))
      {
        getTable().add_value(summary.first, summary.second);
        getTable().set_scientific(summary.first, true);
      }
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
