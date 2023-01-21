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


#include <deal.II/base/flow_function.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <base/global.h>
#include <base/linear_algebra.h>
#include <base/log.h>
#include <factory.h>
#include <stokes/matrixfree/block_schur_preconditioner.h>
#include <stokes/matrixfree/a_block_operator.h>
#include <stokes/matrixfree/schur_block_operator.h>
#include <stokes/matrixfree/stokes_operator.h>
#include <stokes/matrixfree/problem.h>

// #include <ctime>
// #include <iomanip>
// #include <sstream>

using namespace dealii;


namespace
{
  template <int dim, typename LinearAlgebra, int spacedim = dim, typename... Args>
  std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>>
  create_operator(const std::string type, Args &&...args)
  {
    Assert(false, ExcNotImplemented());
  }
} // namespace



namespace StokesMatrixFree
{
  template <int dim, typename LinearAlgebra, int spacedim>
  Problem<dim, LinearAlgebra, spacedim>::Problem(const Parameter &prm)
    : mpi_communicator(MPI_COMM_WORLD)
    , prm(prm)
    // , triangulation(mpi_communicator)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , dof_handler(triangulation)
    , velocities(0)
    , pressure(dim)
  {
    //
    // TODO!!!
    // nearly identical to Poisson
    //

    TimerOutput::Scope t(getTimer(), "initialize_problem");

    // prepare name for logfile
    {
      time_t             now = time(nullptr);
      tm                *ltm = localtime(&now);
      std::ostringstream oss;
      oss << prm.file_stem << "-" << std::put_time(ltm, "%Y%m%d-%H%M%S") << ".log";
      filename_log = oss.str();
    }

    // prepare collections
    // TODO: different mapping for curved cells?
    mapping_collection.push_back(MappingQ1<dim, spacedim>());

    Assert(prm.prm_adaptation.min_p_degree > 1,
           ExcMessage("The minimal polynomial degree must be at least 2!"));

    for (unsigned int degree = 2; degree <= prm.prm_adaptation.max_p_degree; ++degree)
      {
        fe_collection.push_back(FESystem<dim, spacedim>(
          FE_Q<dim, spacedim>(degree), dim, FE_Q<dim, spacedim>(degree - 1), 1));
        quadrature_collection.push_back(QGauss<dim>(degree + 1));
        quadrature_collection_for_errors.push_back(QGauss<dim>(degree + 2));
      }

    const unsigned int min_fe_index = prm.prm_adaptation.min_p_degree - 2;
    fe_collection.set_hierarchy(
      /*next_index=*/
      [](const typename hp::FECollection<dim> &fe_collection,
         const unsigned int                    fe_index) -> unsigned int {
        return ((fe_index + 1) < fe_collection.size()) ? fe_index + 1 : fe_index;
      },
      /*previous_index=*/
      [min_fe_index](const typename hp::FECollection<dim> &,
                     const unsigned int fe_index) -> unsigned int {
        Assert(fe_index >= min_fe_index, ExcMessage("Finite element is not part of hierarchy!"));
        return (fe_index > min_fe_index) ? fe_index - 1 : fe_index;
      });

    // prepare operator (and fe values)
    if (prm.operator_type == "MatrixBased")
      {
        // TODO: make pretty/ remove create_block_operator
        // stokes_operator = std::make_unique<StokesMatrixFree::StokesOperator<dim, LinearAlgebra, spacedim>>(mapping_collection, quadrature_collection);

        a_block_operator =
          std::make_unique<StokesMatrixFree::ABlockOperator<dim, LinearAlgebra, spacedim>>(
            mapping_collection, quadrature_collection, fe_collection);

        schur_block_operator =
          std::make_unique<StokesMatrixFree::SchurBlockOperator<dim, LinearAlgebra, spacedim>>(
            mapping_collection, quadrature_collection, fe_collection);

        // TODO: build only in matrixbased an pass to operators (from above)?
        {
          TimerOutput::Scope t(getTimer(), "calculate_fevalues");

          fe_values_collection =
            std::make_unique<hp::FEValues<dim, spacedim>>(mapping_collection,
                                                          fe_collection,
                                                          quadrature_collection,
                                                          update_values | update_gradients |
                                                            update_quadrature_points |
                                                            update_JxW_values);
          fe_values_collection->precalculate_fe_values();
        }

        // TODO: create operator here
        //       once we move away from the classical matrix based approach
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    // choose functions
    if (prm.grid_type == "kovasznay")
      {
        // boundary_function = Factory::create_function<dim>("zero");
        solution_function = Factory::create_function<dim>("kovasznay exact");
        rhs_function      = Factory::create_function<dim>("kovasznay rhs");
      }
    else if (prm.grid_type == "y-pipe")
      {
        // boundary_function = Factory::create_function<dim>("zero");
        // solution_function = Factory::create_function<dim>("zero");
        // rhs_function      = Factory::create_function<dim>("zero");

        solution_function = std::make_unique<dealii::Functions::ZeroFunction<dim>>(
          /*n_components=*/dim + 1);
        rhs_function = std::make_unique<dealii::Functions::ZeroFunction<dim>>(
          /*n_components=*/dim + 1);
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    // choose adaptation strategy
    adaptation_strategy =
      Factory::create_adaptation<dim, typename LinearAlgebra::BlockVector, spacedim>(
        prm.adaptation_type,
        prm.prm_adaptation,
        locally_relevant_solution,
        fe_collection,
        dof_handler,
        triangulation,
        fe_collection.component_mask(pressure));
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::initialize_grid()
  {
    //
    // TODO!!!
    // nearly identical to Poisson
    //

    TimerOutput::Scope t(getTimer(), "initialize_grid");

    Factory::create_grid(prm.grid_type, triangulation);

    if (prm.resume_filename.compare("") != 0)
      {
        resume_from_checkpoint();
      }
    else
      {
        const unsigned int min_fe_index = prm.prm_adaptation.min_p_degree - 2;
        for (const auto &cell : dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_active_fe_index(min_fe_index);

        dof_handler.distribute_dofs(fe_collection);

        triangulation.refine_global(adaptation_strategy->get_n_initial_refinements());
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::setup_system()
  {
    TimerOutput::Scope t(getTimer(), "setup_system");

    std::vector<unsigned int> stokes_sub_blocks(dim + 1, 0);
    stokes_sub_blocks[dim] = 1;

    {
      TimerOutput::Scope t(getTimer(), "distribute_dofs");

      dof_handler.distribute_dofs(fe_collection);
      DoFRenumbering::component_wise(dof_handler, stokes_sub_blocks);
      partitioning.reinit(dof_handler, stokes_sub_blocks);
    }

    {
      TimerOutput::Scope(getTimer(), "reinit_vectors");

      locally_relevant_solution.reinit(partitioning.get_owned_dofs_per_block(),
                                       partitioning.get_relevant_dofs_per_block(),
                                       mpi_communicator);

      // TODO: remove
      if constexpr (std::is_same_v<typename LinearAlgebra::BlockVector,
                                   dealii::LinearAlgebra::distributed::BlockVector<double>>)
        {
          system_rhs.reinit(partitioning.get_owned_dofs_per_block(),
                            partitioning.get_relevant_dofs_per_block(),
                            mpi_communicator);
        }
      else
        {
          system_rhs.reinit(partitioning.get_owned_dofs_per_block(), mpi_communicator);
        }
    }

    {
      TimerOutput::Scope t(getTimer(), "make_constraints");

      constraints.clear();
      constraints.reinit(partitioning.get_relevant_dofs());

      DoFTools::make_hanging_node_constraints(dof_handler, constraints);

      // TODO: introduce boundary_function
      if (prm.grid_type == "kovasznay")
        {
          VectorTools::interpolate_boundary_values(mapping_collection,
                                                   dof_handler,
                                                   /*boundary_component=*/0,
                                                   *solution_function,
                                                   constraints,
                                                   fe_collection.component_mask(velocities));
        }
      else if (prm.grid_type == "y-pipe")
        {
          Functions::PoisseuilleFlow<dim> inflow(/*radius=*/1.,
                                                 /*Reynolds=*/1.);
          Functions::ZeroFunction<dim>    zero(/*n_components=*/dim + 1);

          // flow at inlet opening 0
          // no slip on walls
          VectorTools::interpolate_boundary_values(mapping_collection,
                                                   dof_handler,
                                                   /*function_map=*/{{0, &inflow}, {3, &zero}},
                                                   constraints,
                                                   fe_collection.component_mask(velocities));
        }
      else
        {
          Assert(false, ExcNotImplemented());
        }

#if false
      // Disable consistency check for now.
      //   see also: https://github.com/dealii/dealii/issues/6255
#  ifdef DEBUG
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
#  endif
#endif
      constraints.close();
    }

    Log::log_hp_diagnostics(triangulation, dof_handler, constraints);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::solve()
  {
    TimerOutput::Scope t(getTimer(), "solve");

    // We need to introduce a vector that does not contain all ghost elements.
    typename LinearAlgebra::BlockVector completely_distributed_solution;
    stokes_operator->initialize_dof_vector(completely_distributed_solution);

    SolverControl solver_control_refined(system_rhs.size(), 1e-8 * system_rhs.l2_norm());

    if (prm.solver_type == "AMG")
      {
        // solve_amg<dim, LinearAlgebra, spacedim>(solver_control_refined,
        //                                         *stokes_operator,
        //                                         *a_block_operator,
        //                                         *schur_block_operator,
        //                                         completely_distributed_solution,
        //                                         system_rhs,
        //                                         dof_handler);
      }
    else if (prm.solver_type == "GMG")
      {
        AssertThrow(false, ExcNotImplemented());
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }

    Log::log_iterations(solver_control_refined);

    constraints.distribute(completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
    locally_relevant_solution.update_ghost_values();



    // TODO: Is this step necessary?
    //       We subtract mean pressure here
    //       no, only necessary for kovasznay
    if (prm.grid_type == "kovasznay")
      {
        const double mean_pressure =
          VectorTools::compute_mean_value(mapping_collection,
                                          dof_handler,
                                          quadrature_collection_for_errors,
                                          locally_relevant_solution,
                                          dim);
        completely_distributed_solution.block(1).add(-mean_pressure);
        locally_relevant_solution.block(1) = completely_distributed_solution.block(1);
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::compute_errors()
  {
    TimerOutput::Scope t(getTimer(), "compute_errors");

    Vector<float> difference_per_cell(triangulation.n_active_cells());

    // QGauss<dim>   quadrature(velocity_degree + 2);
    // todo: why +2 and not +1????
    //       because it is pressure degree + 1 and they re-use these?

    // velocity
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim + 1);
    VectorTools::integrate_difference(dof_handler,
                                      locally_relevant_solution,
                                      *solution_function,
                                      difference_per_cell,
                                      quadrature_collection_for_errors,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    const double L2_error_u =
      VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);

    /*
    VectorTools::integrate_difference(dof_handler,
                                      locally_relevant_solution,
                                      *solution_function,
                                      difference_per_cell,
                                      quadrature_collection_for_errors,
                                      VectorTools::H1_norm,
                                      &velocity_mask);
    const double H1_error_u =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_norm);
    */

    // pressure
    const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
    VectorTools::integrate_difference(dof_handler,
                                      locally_relevant_solution,
                                      *solution_function,
                                      difference_per_cell,
                                      quadrature_collection_for_errors,
                                      VectorTools::L2_norm,
                                      &pressure_mask);
    const double L2_error_p =
      VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);

    /*
    VectorTools::integrate_difference(dof_handler,
                                      locally_relevant_solution,
                                      *solution_function,
                                      difference_per_cell,
                                      quadrature_collection_for_errors,
                                      VectorTools::H1_norm,
                                      &pressure_mask);
    const double H1_error_p =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_norm);
    */

    getPCOut() << "   L2 error velocity:            " << L2_error_u
               << std::endl
               // << "   H1 error velocity:            " << H1_error_u << std::endl
               << "   L2 error pressure:            " << L2_error_p << std::endl;
    // << "   H1 error pressure:            " << H1_error_p << std::endl;

    TableHandler &table = getTable();
    table.add_value("L2_u", L2_error_u);
    // table.add_value("H1_u", H1_error_u);
    table.set_scientific("L2_u", true);
    // table.set_scientific("H1_u", true);

    table.add_value("L2_p", L2_error_p);
    // table.add_value("H1_p", H1_error_p);
    table.set_scientific("L2_p", true);
    // table.set_scientific("H1_p", true);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::output_results()
  {
    TimerOutput::Scope t(getTimer(), "output_results");

    Vector<float> fe_degrees(triangulation.n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        fe_degrees(cell->active_cell_index()) = cell->get_fe().degree;

    Vector<float> subdomain(triangulation.n_active_cells());
    for (auto &subd : subdomain)
      subd = triangulation.locally_owned_subdomain();

    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.add_data_vector(fe_degrees, "fe_degree");
    data_out.add_data_vector(subdomain, "subdomain");

    if (adaptation_strategy->get_error_estimates().size() > 0)
      data_out.add_data_vector(adaptation_strategy->get_error_estimates(), "error");
    if (adaptation_strategy->get_hp_indicators().size() > 0)
      data_out.add_data_vector(adaptation_strategy->get_hp_indicators(), "hp_indicator");

    // TODO: Placeholder for interpolation of correct solution?
    //       Is this necessary???
    /*
    LA::MPI::BlockVector interpolated;
    interpolated.reinit(owned_partitioning, MPI_COMM_WORLD);
    VectorTools::interpolate(dof_handler, ExactSolution<dim>(), interpolated);
    LA::MPI::BlockVector interpolated_relevant(owned_partitioning,
                                               relevant_partitioning,
                                               MPI_COMM_WORLD);
    interpolated_relevant = interpolated;
    {
      std::vector<std::string> solution_names(dim, "ref_u");
      solution_names.emplace_back("ref_p");
      data_out.add_data_vector(interpolated_relevant,
                               solution_names,
                               DataOut<dim>::type_dof_data,
                               data_component_interpretation);
    }
    */

    data_out.build_patches(mapping_collection);

    data_out.write_vtu_with_pvtu_record("./", prm.file_stem, cycle, mpi_communicator, 2, 1);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::resume_from_checkpoint()
  {
    // TODO: same as Poisson

    triangulation.load(prm.resume_filename);

    // custom repartitioning using DoFs requires correctly assigned FEs
    dof_handler.deserialize_active_fe_indices();
    dof_handler.distribute_dofs(fe_collection);
    triangulation.repartition();

    // unpack after repartitioning to avoid unnecessary data transfer
    adaptation_strategy->unpack_after_serialization();

    // load metadata
    std::ifstream                   file(prm.resume_filename + ".metadata", std::ios::binary);
    boost::archive::binary_iarchive ia(file);
    ia >> cycle;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::write_to_checkpoint()
  {
    // TODO: same as Poisson

    // write triangulation and data
    dof_handler.prepare_for_serialization_of_active_fe_indices();
    adaptation_strategy->prepare_for_serialization();

    const std::string filename = prm.file_stem + "-checkpoint";
    triangulation.save(filename);

    // write metadata
    std::ofstream                   file(filename + ".metadata", std::ios::binary);
    boost::archive::binary_oarchive oa(file);
    oa << cycle;

    getPCOut() << "Checkpoint written." << std::endl;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::run()
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

              if ((prm.checkpoint_frequency > 0) && (cycle % prm.checkpoint_frequency == 0))
                write_to_checkpoint();
            }

          Log::log_cycle(cycle, prm);

          setup_system();

          // TODO: I am not happy with this
          if (prm.operator_type == "MatrixBased")
            {
              // stokes_operator->reinit(
              //   partitioning, dof_handler, constraints, system_rhs, rhs_function.get());

              if (prm.operator_type == "MatrixBased")
                Log::log_nonzero_elements(stokes_operator->get_system_matrix());

              a_block_operator->reinit(partitioning, dof_handler, constraints);
              schur_block_operator->reinit(partitioning, dof_handler, constraints);

              solve();
            }
          else
            {
              Assert(false, ExcInternalError());
            }

          // compute_errors();
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



#ifdef DEAL_II_WITH_TRILINOS
  template class Problem<2, dealiiTrilinos, 2>;
  template class Problem<3, dealiiTrilinos, 3>;
  template class Problem<2, Trilinos, 2>;
  template class Problem<3, Trilinos, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
  template class Problem<2, PETSc, 2>;
  template class Problem<3, PETSc, 3>;
#endif

} // namespace Stokes
