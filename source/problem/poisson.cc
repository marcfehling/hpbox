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


#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <adaptation/factory.h>
#include <base/global.h>
#include <base/linear_algebra.h>
#include <function/factory.h>
#include <grid/factory.h>
#include <log/log.h>
#include <problem/poisson.h>
#include <solver/cg/amg.h>
#include <solver/cg/gmg.h>

#include <ctime>
#include <iomanip>
#include <sstream>

using namespace dealii;


namespace Problem
{
  template <int dim, typename LinearAlgebra, int spacedim>
  Poisson<dim, LinearAlgebra, spacedim>::Poisson(const Parameters &prm)
    : mpi_communicator(MPI_COMM_WORLD)
    , prm(prm)
    , triangulation(mpi_communicator)
    , dof_handler(triangulation)
  {
    TimerOutput::Scope t(getTimer(), "initialize_problem");

    // prepare name for logfile
    {
      time_t             now = time(nullptr);
      tm                *ltm = localtime(&now);
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
    // TODO: Maybe move this to separate factory function?
    if (prm.operator_type == "MatrixBased")
      {
        {
          TimerOutput::Scope t(getTimer(), "calculate_fevalues");

          fe_values_collection = std::make_unique<hp::FEValues<dim, spacedim>>(
            mapping_collection,
            fe_collection,
            quadrature_collection,
            update_values | update_gradients | update_quadrature_points |
              update_JxW_values);
          fe_values_collection->precalculate_fe_values();
        }

        poisson_operator_matrixbased = std::make_unique<
          Operator::Poisson::MatrixBased<dim, LinearAlgebra, spacedim>>(
          mapping_collection, quadrature_collection, *fe_values_collection);
      }
    else if (prm.operator_type == "MatrixFree")
      {
        if constexpr (std::is_same<LinearAlgebra, dealiiTrilinos>::value)
          poisson_operator_matrixfree = std::make_unique<
            Operator::Poisson::MatrixFree<dim, LinearAlgebra, spacedim>>(
            mapping_collection, quadrature_collection);
        else
          Assert(false,
                 ExcMessage(
                   "MatrixFree only availble with dealii & Trilinos!"));
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    // choose functions
    boundary_function = Factory::create_function<dim>("reentrant corner");
    solution_function = Factory::create_function<dim>("reentrant corner");
    // rhs_function      = Factory::create_function<dim>("zero");

    // choose adaptation strategy
    adaptation_strategy =
      Factory::create_adaptation<dim, typename LinearAlgebra::Vector, spacedim>(
        prm.adaptation_type,
        prm.prm_adaptation,
        locally_relevant_solution,
        fe_collection,
        dof_handler,
        triangulation);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Poisson<dim, LinearAlgebra, spacedim>::initialize_grid()
  {
    TimerOutput::Scope t(getTimer(), "initialize_grid");

    Factory::create_grid("reentrant corner", triangulation);

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
  Poisson<dim, LinearAlgebra, spacedim>::setup_system()
  {
    TimerOutput::Scope t(getTimer(), "setup");

    dof_handler.distribute_dofs(fe_collection);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(
      mapping_collection, dof_handler, 0, *boundary_function, constraints);

#ifdef DEBUG
    // We have not dealt with chains of constraints on ghost cells yet.
    // Thus, we are content with verifying their consistency for now.
    const std::vector<IndexSet> locally_owned_dofs_per_processor =
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
  Poisson<dim, LinearAlgebra, spacedim>::solve(
    const OperatorType                   &system_matrix,
    typename LinearAlgebra::Vector       &locally_relevant_solution,
    const typename LinearAlgebra::Vector &system_rhs)
  {
    TimerOutput::Scope t(getTimer(), "solve");

    typename LinearAlgebra::Vector completely_distributed_solution;
    typename LinearAlgebra::Vector completely_distributed_system_rhs;

    if constexpr (std::is_same<
                    typename LinearAlgebra::Vector,
                    dealii::LinearAlgebra::distributed::Vector<double>>::value)
      {
        system_matrix.initialize_dof_vector(completely_distributed_solution);
        system_matrix.initialize_dof_vector(completely_distributed_system_rhs);

        completely_distributed_system_rhs.copy_locally_owned_data_from(
          system_rhs);
      }
    else
      {
        completely_distributed_solution.reinit(locally_owned_dofs,
                                               mpi_communicator);
        completely_distributed_system_rhs = system_rhs;
      }

    SolverControl solver_control(completely_distributed_system_rhs.size(),
                                 1e-12 *
                                   completely_distributed_system_rhs.l2_norm());

    if (prm.solver_type == "AMG")
      {
        Solver::CG::AMG::solve<LinearAlgebra>(
          solver_control,
          system_matrix,
          completely_distributed_solution,
          completely_distributed_system_rhs);
      }
    else if (prm.solver_type == "GMG")
      {
        if constexpr (std::is_same<
                        OperatorType,
                        Operator::Poisson::
                          MatrixFree<dim, dealiiTrilinos, spacedim>>::value)
          {
            Solver::CG::GMG::solve(solver_control,
                                   system_matrix,
                                   completely_distributed_solution,
                                   completely_distributed_system_rhs,
                                   /*boundary_values=*/mapping_collection,
                                   dof_handler,
                                   /*operator_constructor=*/mapping_collection,
                                   quadrature_collection);
          }
        else if constexpr (std::is_same<
                             OperatorType,
                             Operator::Poisson::MatrixBased<dim,
                                                            dealiiTrilinos,
                                                            spacedim>>::value)
          {
            Solver::CG::GMG::solve(solver_control,
                                   system_matrix,
                                   completely_distributed_solution,
                                   completely_distributed_system_rhs,
                                   /*boundary_values=*/mapping_collection,
                                   dof_handler,
                                   /*operator_constructor=*/mapping_collection,
                                   quadrature_collection,
                                   *fe_values_collection);
          }
        else
          {
            Assert(false,
                   ExcMessage("GMG is only available with dealii & Trilinos!"));
          }
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    getPCOut() << "   Number of iterations:         "
               << solver_control.last_step() << std::endl;
    getTable().add_value("iteratations", solver_control.last_step());

    constraints.distribute(completely_distributed_solution);

    if constexpr (std::is_same<
                    typename LinearAlgebra::Vector,
                    dealii::LinearAlgebra::distributed::Vector<double>>::value)
      {
        locally_relevant_solution.copy_locally_owned_data_from(
          completely_distributed_solution);
        locally_relevant_solution.update_ghost_values();
      }
    else
      {
        locally_relevant_solution = completely_distributed_solution;
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Poisson<dim, LinearAlgebra, spacedim>::compute_errors()
  {
    TimerOutput::Scope t(getTimer(), "compute_errors");

    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      locally_relevant_solution,
                                      *solution_function,
                                      difference_per_cell,
                                      quadrature_collection,
                                      VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      locally_relevant_solution,
                                      *solution_function,
                                      difference_per_cell,
                                      quadrature_collection,
                                      VectorTools::H1_norm);
    const double H1_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_norm);

    getPCOut() << "   L2 error:                     " << L2_error << std::endl
               << "   H1 error:                     " << H1_error << std::endl;

    TableHandler &table = getTable();
    table.add_value("L2", L2_error);
    table.add_value("H1", H1_error);
    table.set_scientific("L2", true);
    table.set_scientific("H1", true);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Poisson<dim, LinearAlgebra, spacedim>::output_results()
  {
    TimerOutput::Scope t(getTimer(), "output_results");

    Vector<float> fe_degrees(triangulation.n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        fe_degrees(cell->active_cell_index()) = cell->get_fe().degree;

    Vector<float> subdomain(triangulation.n_active_cells());
    for (auto &subd : subdomain)
      subd = triangulation.locally_owned_subdomain();

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(locally_relevant_solution, "solution");
    data_out.add_data_vector(fe_degrees, "fe_degree");
    data_out.add_data_vector(subdomain, "subdomain");

    if (adaptation_strategy->get_error_estimates().size() > 0)
      data_out.add_data_vector(adaptation_strategy->get_error_estimates(),
                               "error");
    if (adaptation_strategy->get_hp_indicators().size() > 0)
      data_out.add_data_vector(adaptation_strategy->get_hp_indicators(),
                               "hp_indicator");

    data_out.build_patches(mapping_collection);

    data_out.write_vtu_with_pvtu_record(
      "./", prm.file_stem, cycle, mpi_communicator, 2, 1);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Poisson<dim, LinearAlgebra, spacedim>::resume_from_checkpoint()
  {
    triangulation.load(prm.resume_filename);

    // custom repartitioning using DoFs requires correctly assigned FEs
    dof_handler.deserialize_active_fe_indices();
    dof_handler.distribute_dofs(fe_collection);
    triangulation.repartition();

    // unpack after repartitioning to avoid unnecessary data transfer
    adaptation_strategy->unpack_after_serialization();

    // load metadata
    std::ifstream file(prm.resume_filename + ".metadata", std::ios::binary);
    boost::archive::binary_iarchive ia(file);
    ia >> cycle;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Poisson<dim, LinearAlgebra, spacedim>::write_to_checkpoint()
  {
    // write triangulation and data
    dof_handler.prepare_for_serialization_of_active_fe_indices();
    adaptation_strategy->prepare_for_serialization();

    const std::string filename = prm.file_stem + "-checkpoint";
    triangulation.save(filename);

    // write metadata
    std::ofstream file(filename + ".metadata", std::ios::binary);
    boost::archive::binary_oarchive oa(file);
    oa << cycle;

    getPCOut() << "Checkpoint written." << std::endl;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Poisson<dim, LinearAlgebra, spacedim>::run()
  {
    getTable().set_auto_fill_mode(true);

    for (cycle = 0; cycle < adaptation_strategy->get_n_cycles(); ++cycle)
      {
        {
          TimerOutput::Scope t(getTimer(), "full_cycle");

          if (cycle == 0)
            {
              initialize_grid();
            }
          else
            {
              adaptation_strategy->refine();

              if ((prm.checkpoint_frequency > 0) &&
                  (cycle % prm.checkpoint_frequency == 0))
                write_to_checkpoint();
            }

          getPCOut() << "Cycle " << cycle << ':' << std::endl;
          getTable().add_value("cycle", cycle);

          setup_system();

          Log::log_hp_diagnostics(triangulation, dof_handler, constraints);

          // TODO: I am not happy with this
          if (prm.operator_type == "MatrixBased")
            {
              poisson_operator_matrixbased->reinit(dof_handler,
                                                   constraints,
                                                   system_rhs);
              solve(*poisson_operator_matrixbased,
                    locally_relevant_solution,
                    system_rhs);
            }
          else if (prm.operator_type == "MatrixFree")
            {
              poisson_operator_matrixfree->reinit(dof_handler,
                                                  constraints,
                                                  system_rhs);
              solve(*poisson_operator_matrixfree,
                    locally_relevant_solution,
                    system_rhs);
            }
          else
            {
              Assert(false, ExcInternalError());
            }

          compute_errors();
          adaptation_strategy->estimate_mark();

          if ((prm.output_frequency > 0) && (cycle % prm.output_frequency == 0))
            output_results();
        }

        Log::log_timings();

        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          {
            std::ofstream logstream(filename_log);
            getTable().write_text(logstream);
          }

        getTimer().reset();
        getTable().start_new_row();
      }
  }



  // explicit instantiations
#ifdef DEAL_II_WITH_TRILINOS
  template class Poisson<2, dealiiTrilinos, 2>;
  template class Poisson<3, dealiiTrilinos, 3>;
  template class Poisson<2, Trilinos, 2>;
  template class Poisson<3, Trilinos, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
  template class Poisson<2, PETSc, 2>;
  template class Poisson<3, PETSc, 3>;
#endif

} // namespace Problem
