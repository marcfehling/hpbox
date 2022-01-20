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


#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <adaptation/factory.h>
#include <base/global.h>
#include <base/linear_algebra.h>
#include <grid/factory.h>
#include <log/log.h>
#include <problem/stokes.h>
#include <solver/cg/amg.h>
#include <solver/cg/gmg.h>

//#include <ctime>
//#include <iomanip>
//#include <sstream>

using namespace dealii;


namespace Problem
{
  template <int dim, typename LinearAlgebra, int spacedim>
  Stokes<dim, LinearAlgebra, spacedim>::Stokes(const Parameters &prm)
    : mpi_communicator(MPI_COMM_WORLD)
    , prm(prm)
    , triangulation(mpi_communicator)
    , dof_handler(triangulation)
  {
    //
    // TODO!!!
    // nearly identical to Poisson
    //

    TimerOutput::Scope t(getTimer(), "initialize_problem");

    // prepare name for logfile
    {
      time_t             now = time(nullptr);
      tm *               ltm = localtime(&now);
      std::ostringstream oss;
      oss << prm.file_stem << "-" << std::put_time(ltm, "%Y%m%d-%H%M%S")
          << ".log";
      filename_log = oss.str();
    }

    // prepare collections
    mapping_collection.push_back(MappingQ1<dim, spacedim>());

    for (unsigned int degree = 1; degree <= prm.prm_adaptation.max_p_degree;
         ++degree)
      {
        fe_collection.push_back(FE_Q<dim, spacedim>(degree));
        quadrature_collection.push_back(QGauss<dim>(degree + 1));
      }

    const unsigned int min_fe_index = prm.prm_adaptation.min_p_degree - 1;
    fe_collection.set_hierarchy(
      /*next_index=*/
      [](const typename hp::FECollection<dim> &fe_collection,
         const unsigned int                    fe_index) -> unsigned int {
        return ((fe_index + 1) < fe_collection.size()) ? fe_index + 1 :
                                                         fe_index;
      },
      /*previous_index=*/
      [min_fe_index](const typename hp::FECollection<dim> &,
                     const unsigned int fe_index) -> unsigned int {
        Assert(fe_index >= min_fe_index,
               ExcMessage("Finite element is not part of hierarchy!"));
        return (fe_index > min_fe_index) ? fe_index - 1 : fe_index;
      });

    // prepare operator (and fe values)
    if (prm.operator_type == "MatrixBased")
      {

      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    // choose functions

    // choose adaptation strategy
    adaptation_strategy =
      Factory::create_adaptation<dim, typename LinearAlgebra::BlockVector, spacedim>(
        prm.adaptation_type,
        prm.prm_adaptation,
        locally_relevant_solution,
        fe_collection,
        dof_handler,
        triangulation);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Stokes<dim, LinearAlgebra, spacedim>::initialize_grid()
  {
    //
    // TODO!!!
    // nearly identical to Poisson
    //

    TimerOutput::Scope t(getTimer(), "initialize_grid");

    // Factory::create_grid("y-pipe", triangulation);
    // first reproduce step-22
    {
      std::vector<unsigned int> subdivisions(dim, 1);
      subdivisions[0] = 4;
      const Point<dim> bottom_left = (dim == 2 ?
                                        Point<dim>(-2, -1) :    // 2d case
                                        Point<dim>(-2, 0, -1)); // 3d case
      const Point<dim> top_right = (dim == 2 ?
                                      Point<dim>(2, 0) :    // 2d case
                                      Point<dim>(2, 1, 0)); // 3d case
      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                subdivisions,
                                                bottom_left,
                                                top_right);
    }

    if (prm.resume_filename.compare("") != 0)
      {
        resume_from_checkpoint();
      }
    else
      {
        const unsigned int min_fe_index = prm.prm_adaptation.min_p_degree - 1;
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_active_fe_index(min_fe_index);

        triangulation.refine_global(
          adaptation_strategy->get_n_initial_refinements());
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Stokes<dim, LinearAlgebra, spacedim>::setup_system()
  {
    TimerOutput::Scope t(getTimer(), "setup");

    dof_handler.distribute_dofs(fe_collection);
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    // VectorTools::interpolate_boundary_values(
    //   mapping_collection, dof_handler, 0, *boundary_function, constraints);

#ifdef DEBUG
    // We have not dealt with chains of constraints on ghost cells yet.
    // Thus, we are content with verifying their consistency for now.
    std::vector<IndexSet> locally_owned_dofs_per_processor =
      Utilities::MPI::all_gather(mpi_communicator,
                                 dof_handler.locally_owned_dofs());

    IndexSet locally_active_dofs;
    DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);

    AssertThrow(
      constraints.is_consistent_in_parallel(locally_owned_dofs_per_processor,
                                            locally_active_dofs,
                                            mpi_communicator,
                                            /*verbose=*/true),
      ExcMessage("AffineConstraints object contains inconsistencies!"));
#endif
    constraints.close();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  template <typename OperatorType>
  void
  Stokes<dim, LinearAlgebra, spacedim>::solve(
    const OperatorType &                  system_matrix,
    typename LinearAlgebra::Vector &      locally_relevant_solution,
    const typename LinearAlgebra::Vector &system_rhs)
  {
    TimerOutput::Scope t(getTimer(), "solve");
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Stokes<dim, LinearAlgebra, spacedim>::compute_errors()
  {
    TimerOutput::Scope t(getTimer(), "compute_errors");
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Stokes<dim, LinearAlgebra, spacedim>::output_results()
  {
    TimerOutput::Scope t(getTimer(), "output_results");
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Stokes<dim, LinearAlgebra, spacedim>::resume_from_checkpoint()
  {
    triangulation.load(prm.resume_filename);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Stokes<dim, LinearAlgebra, spacedim>::write_to_checkpoint()
  {
    // write triangulation and data
    dof_handler.prepare_for_serialization_of_active_fe_indices();
    adaptation_strategy->prepare_for_serialization();

    const std::string filename = prm.file_stem + "-checkpoint";
    triangulation.save(filename);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Stokes<dim, LinearAlgebra, spacedim>::run()
  {
    getTable().set_auto_fill_mode(true);
  }



#ifdef DEAL_II_WITH_TRILINOS
    template class Stokes<2, dealiiTrilinos, 2>;
    template class Stokes<3, dealiiTrilinos, 3>;
    template class Stokes<2, Trilinos, 2>;
    template class Stokes<3, Trilinos, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
    template class Stokes<2, PETSc, 2>;
    template class Stokes<3, PETSc, 3>;
#endif

} // namespace Problem
