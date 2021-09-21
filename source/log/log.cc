// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
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
#include <log/log.h>

#include <algorithm>
#include <vector>

using namespace dealii;


namespace Log
{
  void
  log_timings()
  {
    getTimer().print_summary();
    getPCOut() << std::endl;

    for (const auto &summary :
         getTimer().get_summary_data(TimerOutput::total_wall_time))
      {
        getTable().add_value(summary.first, summary.second);
        getTable().set_scientific(summary.first, true);
      }
  }



  template <int dim, typename T, int spacedim>
  void
  log_hp_diagnostics(
    const parallel::distributed::Triangulation<dim, spacedim> &triangulation,
    const DoFHandler<dim, spacedim> &                          dof_handler,
    const AffineConstraints<T> &                               constraints)
  {
    ConditionalOStream &pcout = getPCOut();
    TableHandler &      table = getTable();

    const MPI_Comm &mpi_communicator = dof_handler.get_communicator();
    const hp::FECollection<dim, spacedim> &fe_collection =
      dof_handler.get_fe_collection();

    const unsigned int first_n_processes =
      std::min<unsigned int>(8,
                             Utilities::MPI::n_mpi_processes(mpi_communicator));
    const bool output_cropped =
      first_n_processes < Utilities::MPI::n_mpi_processes(mpi_communicator);

    {
      pcout << "   Number of active cells:       "
            << triangulation.n_global_active_cells() << std::endl;
      table.add_value("active_cells", triangulation.n_global_active_cells());

      pcout << "     by partition:              ";
      std::vector<unsigned int> n_active_cells_per_subdomain =
        Utilities::MPI::gather(mpi_communicator,
                               triangulation.n_locally_owned_active_cells());
      for (unsigned int i = 0; i < first_n_processes; ++i)
        pcout << ' ' << n_active_cells_per_subdomain[i];
      if (output_cropped)
        pcout << " ...";
      pcout << std::endl;
    }

    {
      std::vector<unsigned int> n_fe_indices(fe_collection.size(), 0);
      for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          n_fe_indices[cell->active_fe_index()]++;

      Utilities::MPI::sum(n_fe_indices, mpi_communicator, n_fe_indices);

      pcout << "   Frequencies of poly. degrees:";
      for (unsigned int i = 0; i < fe_collection.size(); ++i)
        if (n_fe_indices[i] > 0)
          pcout << ' ' << fe_collection[i].degree << ":" << n_fe_indices[i];
      pcout << std::endl;
    }

    {
      pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
      table.add_value("dofs", dof_handler.n_dofs());

      pcout << "     by partition:              ";
      std::vector<types::global_dof_index> n_dofs_per_subdomain =
        Utilities::MPI::gather(mpi_communicator,
                               dof_handler.n_locally_owned_dofs());
      for (unsigned int i = 0; i < first_n_processes; ++i)
        pcout << ' ' << n_dofs_per_subdomain[i];
      if (output_cropped)
        pcout << " ...";
      pcout << std::endl;
    }

    {
      std::vector<types::global_dof_index> n_constraints_per_subdomain =
        Utilities::MPI::gather(mpi_communicator, constraints.n_constraints());
      const unsigned int n_constraints =
        std::accumulate(n_constraints_per_subdomain.begin(),
                        n_constraints_per_subdomain.end(),
                        0);

      pcout << "   Number of constraints:        " << n_constraints
            << std::endl;
      table.add_value("constraints", n_constraints);

      pcout << "     by partition:              ";
      for (unsigned int i = 0; i < first_n_processes; ++i)
        pcout << ' ' << n_constraints_per_subdomain[i];
      if (output_cropped)
        pcout << " ...";
      pcout << std::endl;
    }

    {
      std::vector<types::global_dof_index> n_identities_per_subdomain =
        Utilities::MPI::gather(mpi_communicator, constraints.n_identities());
      const unsigned int n_identities =
        std::accumulate(n_identities_per_subdomain.begin(),
                        n_identities_per_subdomain.end(),
                        0);

      pcout << "   Number of identities:         " << n_identities << std::endl;
      table.add_value("identities", n_identities);

      pcout << "     by partition:              ";
      for (unsigned int i = 0; i < first_n_processes; ++i)
        pcout << ' ' << n_identities_per_subdomain[i];
      if (output_cropped)
        pcout << " ...";
      pcout << std::endl;

      const float fraction =
        static_cast<float>(n_identities) / dof_handler.n_dofs();
      pcout << "   Fraction of identities:       " << 100 * fraction << "%"
            << std::endl;
    }
  }



  // explicit instantiations
  template void
  log_hp_diagnostics<2, double, 2>(
    const parallel::distributed::Triangulation<2, 2> &,
    const DoFHandler<2, 2> &,
    const AffineConstraints<double> &);
  template void
  log_hp_diagnostics<3, double, 3>(
    const parallel::distributed::Triangulation<3, 3> &,
    const DoFHandler<3, 3> &,
    const AffineConstraints<double> &);
} // namespace Log
