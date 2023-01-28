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
#include <stokes/matrixfree/a_block_operator.h>
#include <stokes/matrixfree/block_schur_preconditioner.h>
#include <stokes/matrixfree/problem.h>
#include <stokes/matrixfree/schur_block_operator.h>
#include <stokes/matrixfree/stokes_operator.h>

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
    , dof_handler_v(triangulation)
    , dof_handler_p(triangulation)
    , dof_handlers({&dof_handler_v, &dof_handler_p})
    , constraints({&constraints_v, &constraints_p})
    , locally_relevant_solution(2)
  {
    AssertThrow(prm.operator_type == "MatrixFree", ExcNotImplemented());

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
        fe_collection_v.push_back(FESystem<dim, spacedim>(FE_Q<dim, spacedim>(degree), dim));
        fe_collection_p.push_back(FE_Q<dim, spacedim>(degree - 1));

        quadrature_collection_v.push_back(QGauss<dim>(degree + 1));
        quadrature_collection_p.push_back(QGauss<dim>(degree + 1)); // TODO: reduce by one!
        quadrature_collections = {quadrature_collection_v, quadrature_collection_p};

        quadrature_collection_for_errors.push_back(QGauss<dim>(degree + 2)); // TODO: reduce by one?
      }

    {
      const unsigned int min_fe_index = prm.prm_adaptation.min_p_degree - 2;

      const auto next_index = [](const typename hp::FECollection<dim> &fe_collection,
                                 const unsigned int                    fe_index) -> unsigned int {
        return ((fe_index + 1) < fe_collection.size()) ? fe_index + 1 : fe_index;
      };
      const auto previous_index = [min_fe_index](const typename hp::FECollection<dim> &,
                                                 const unsigned int fe_index) -> unsigned int {
        Assert(fe_index >= min_fe_index, ExcMessage("Finite element is not part of hierarchy!"));
        return (fe_index > min_fe_index) ? fe_index - 1 : fe_index;
      };

      fe_collection_v.set_hierarchy(next_index, previous_index);
      fe_collection_p.set_hierarchy(next_index, previous_index);
    }

    // prepare operator
    {
      stokes_operator = std::make_unique<StokesMatrixFree::StokesOperator<dim, LinearAlgebra, spacedim>>(mapping_collection, quadrature_collections);

      a_block_operator =
        std::make_unique<StokesMatrixFree::ABlockOperator<dim, LinearAlgebra, spacedim>>(
          mapping_collection, quadrature_collection_v, fe_collection_v);

      schur_block_operator =
        std::make_unique<StokesMatrixFree::SchurBlockOperator<dim, LinearAlgebra, spacedim>>(
          mapping_collection, quadrature_collection_p, fe_collection_p);
    }

    // choose functions
    if (prm.grid_type == "kovasznay")
      {
        // boundary_function = Factory::create_function<dim>("zero");
        solution_function_v = Factory::create_function<dim>("kovasznay exact velocity");
        solution_function_p = Factory::create_function<dim>("kovasznay exact pressure");

        rhs_function_v      = Factory::create_function<dim>("kovasznay rhs velocity");
        // rhs_function_p      = Factory::create_function<dim>("kovasznay rhs pressure");
        rhs_function_p = std::make_shared<Functions::ZeroFunction<dim>>(1);
      }
    else if (prm.grid_type == "y-pipe")
      {
        // boundary_function = Factory::create_function<dim>("zero");
        // solution_function = Factory::create_function<dim>("zero");
        // rhs_function      = Factory::create_function<dim>("zero");

        // solution_function_v = Factory::create_function<dim>("zero", /*n_components=*/dim);
        // solution_function_p = Factory::create_function<dim>("zero", /*n_components=*/1);
        // rhs_function_v = Factory::create_function<dim>("zero", /*n_components=*/dim);
        // rhs_function_p = Factory::create_function<dim>("zero", /*n_components=*/1);

        solution_function_v = std::make_unique<Functions::ZeroFunction<dim>>(dim);
        solution_function_p = std::make_unique<Functions::ZeroFunction<dim>>(1);

        rhs_function_v = std::make_shared<Functions::ZeroFunction<dim>>(dim);
        rhs_function_p = std::make_shared<Functions::ZeroFunction<dim>>(1);
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
    rhs_functions = {rhs_function_v.get(), rhs_function_p.get()};

    // choose adaptation strategy
    adaptation_strategy_p =
      Factory::create_adaptation<dim, typename LinearAlgebra::Vector, spacedim>(
        prm.adaptation_type,
        prm.prm_adaptation,
        locally_relevant_solution.block(1),
        fe_collection_p,
        dof_handler_p,
        triangulation);

    // cell weighting
    if (prm.adaptation_type != "h")
      {
        cell_weights_v.reinit(dof_handler_v,
                              parallel::CellWeights<dim, spacedim>::ndofs_weighting(
                                {prm.prm_adaptation.weighting_factor,
                                 prm.prm_adaptation.weighting_exponent}));
        cell_weights_p.reinit(dof_handler_p,
                              parallel::CellWeights<dim, spacedim>::ndofs_weighting(
                                {prm.prm_adaptation.weighting_factor,
                                 prm.prm_adaptation.weighting_exponent}));
      }
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
        for (const auto &cell : dof_handler_v.active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_active_fe_index(min_fe_index);
        for (const auto &cell : dof_handler_p.active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_active_fe_index(min_fe_index);

        dof_handler_v.distribute_dofs(fe_collection_v);
        dof_handler_p.distribute_dofs(fe_collection_p);

        triangulation.refine_global(adaptation_strategy_p->get_n_initial_refinements());
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

      dof_handler_v.distribute_dofs(fe_collection_v);
      dof_handler_p.distribute_dofs(fe_collection_p);

      partitioning_v.reinit(dof_handler_v);
      partitioning_p.reinit(dof_handler_p);
    }

    {
      TimerOutput::Scope(getTimer(), "reinit_vectors");

      locally_relevant_solution.block(0).reinit(partitioning_v.get_owned_dofs(),
                                                partitioning_v.get_relevant_dofs(),
                                                mpi_communicator);
      locally_relevant_solution.block(1).reinit(partitioning_p.get_owned_dofs(),
                                                partitioning_p.get_relevant_dofs(),
                                                mpi_communicator);
      locally_relevant_solution.collect_sizes();
    }

    {
      TimerOutput::Scope t(getTimer(), "make_constraints");

      {
        constraints_v.clear();
        constraints_v.reinit(partitioning_v.get_relevant_dofs());

        DoFTools::make_hanging_node_constraints(dof_handler_v, constraints_v);

        // TODO: introduce boundary_function
        if (prm.grid_type == "kovasznay")
          {
            VectorTools::interpolate_boundary_values(mapping_collection,
                                                     dof_handler_v,
                                                     /*boundary_component=*/0,
                                                     *solution_function_v,
                                                     constraints_v);
          }
        else if (prm.grid_type == "y-pipe")
          {
            ::Function::PoisseuilleFlowVelocity<dim> inflow(/*radius=*/1.);
            Functions::ZeroFunction<dim>             zero(/*n_components=*/dim);

            // flow at inlet opening 0
            // no slip on walls
            VectorTools::interpolate_boundary_values(mapping_collection,
                                                     dof_handler_v,
                                                     /*function_map=*/{{0, &inflow}, {3, &zero}},
                                                     constraints_v);
          }
        else
          {
            Assert(false, ExcNotImplemented());
          }

        constraints_v.close();
      }

      {
        constraints_p.clear();
        constraints_p.reinit(partitioning_p.get_relevant_dofs());

        DoFTools::make_hanging_node_constraints(dof_handler_p, constraints_p);

        constraints_p.close();
      }
    }

    // TODO: check consistency

    // TODO: also log p
    // add new function to Log that takes vector
    Log::log_hp_diagnostics(triangulation, dof_handler_v, constraints_v);
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

    constraints_v.distribute(completely_distributed_solution.block(0));
    constraints_p.distribute(completely_distributed_solution.block(1));

    locally_relevant_solution = completely_distributed_solution;
    locally_relevant_solution.update_ghost_values();



    // TODO: Is this step necessary?
    //       We subtract mean pressure here
    //       no, only necessary for kovasznay
    if (prm.grid_type == "kovasznay")
      {
        const double mean_pressure =
          VectorTools::compute_mean_value(mapping_collection,
                                          dof_handler_p,
                                          quadrature_collection_for_errors,
                                          locally_relevant_solution.block(1),
                                          /*component=*/0);
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
    VectorTools::integrate_difference(dof_handler_v,
                                      locally_relevant_solution.block(0),
                                      *solution_function_v,
                                      difference_per_cell,
                                      quadrature_collection_for_errors,
                                      VectorTools::L2_norm);
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
    VectorTools::integrate_difference(dof_handler_p,
                                      locally_relevant_solution.block(1),
                                      *solution_function_p,
                                      difference_per_cell,
                                      quadrature_collection_for_errors,
                                      VectorTools::L2_norm);
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
    for (const auto &cell : dof_handler_p.active_cell_iterators())
      if (cell->is_locally_owned())
        fe_degrees(cell->active_cell_index()) = cell->get_fe().degree;

    Vector<float> subdomain(triangulation.n_active_cells());
    for (auto &subd : subdomain)
      subd = triangulation.locally_owned_subdomain();

    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler_v,
                             locally_relevant_solution.block(0),
                             "velocity",
                             data_component_interpretation);
    data_out.add_data_vector(dof_handler_p, locally_relevant_solution.block(1), "pressure");
    //                         DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(dof_handler_p, fe_degrees, "fe_degree");
    data_out.add_data_vector(dof_handler_p, subdomain, "subdomain");

    if (adaptation_strategy_p->get_error_estimates().size() > 0)
      data_out.add_data_vector(dof_handler_p,
                               adaptation_strategy_p->get_error_estimates(),
                               "error");
    if (adaptation_strategy_p->get_hp_indicators().size() > 0)
      data_out.add_data_vector(dof_handler_p,
                               adaptation_strategy_p->get_hp_indicators(),
                               "hp_indicator");

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
    dof_handler_p.deserialize_active_fe_indices();
    dof_handler_v.set_active_fe_indices(dof_handler_p.get_active_fe_indices());

    dof_handler_p.distribute_dofs(fe_collection_p);
    dof_handler_v.distribute_dofs(fe_collection_v);

    triangulation.repartition();

    // unpack after repartitioning to avoid unnecessary data transfer
    adaptation_strategy_p->unpack_after_serialization();

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
    dof_handler_p.prepare_for_serialization_of_active_fe_indices();
    adaptation_strategy_p->prepare_for_serialization();

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

    for (cycle = 0; cycle < adaptation_strategy_p->get_n_cycles(); ++cycle)
      {
        {
          TimerOutput::Scope t(getTimer(), "full_cycle");

          if (cycle == 0)
            {
              initialize_grid();
            }
          else
            {
              adaptation_strategy_p->refine();
              dof_handler_v.set_active_fe_indices(dof_handler_p.get_active_fe_indices());

              if ((prm.checkpoint_frequency > 0) && (cycle % prm.checkpoint_frequency == 0))
                write_to_checkpoint();
            }

#ifdef DEBUG
          // check if both dofhandlers have same fe indices
          std::vector<types::fe_index> fe_indices_v = dof_handler_v.get_active_fe_indices();
          std::vector<types::fe_index> fe_indices_p = dof_handler_p.get_active_fe_indices();
          Assert(std::equal(fe_indices_v.begin(), fe_indices_v.end(), fe_indices_p.begin()),
                 ExcMessage("Active FE indices differ!"));
#endif

          Log::log_cycle(cycle, prm);

          setup_system();

          stokes_operator->reinit(dof_handlers, constraints, system_rhs, rhs_functions);

          a_block_operator->reinit(partitioning_v, dof_handler_v, constraints_v);
          schur_block_operator->reinit(partitioning_p, dof_handler_p, constraints_p);

          solve();

          // compute_errors();
          adaptation_strategy_p->estimate_mark();

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
#endif

} // namespace StokesMatrixFree
