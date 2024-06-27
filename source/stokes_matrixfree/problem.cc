// ---------------------------------------------------------------------
//
// Copyright (C) 2022 - 2023 by the deal.II authors
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

#include <factory.h>
#include <global.h>
#include <linear_algebra.h>
#include <log.h>
#include <stokes_matrixfree/operators.h>
#include <stokes_matrixfree/problem.h>
#include <stokes_matrixfree/solvers.h>

using namespace dealii;



namespace StokesMatrixFree
{
  template <int dim, typename LinearAlgebra, int spacedim>
  Problem<dim, LinearAlgebra, spacedim>::Problem(const Parameter &prm)
    : mpi_communicator(MPI_COMM_WORLD)
    , prm(prm)
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
    TimerOutput::Scope t(getTimer(), "initialize_problem");

    // prepare name for logfile
    filename_stem = prm.file_stem + "-" + prm.logfile_suffix;
    filename_log  = filename_stem + ".log";

    // prepare collections
    // TODO: different mapping for curved cells?
    mapping_collection.push_back(MappingQ1<dim, spacedim>());

    Assert(prm.prm_adaptation.min_p_degree > 1,
           ExcMessage("The minimal polynomial degree must be at least 2!"));

    for (unsigned int degree = 1; degree <= prm.prm_adaptation.max_p_degree; ++degree)
      {
        fe_collection_v.push_back(FESystem<dim, spacedim>(FE_Q<dim, spacedim>(degree), dim));
        quadrature_collection_v.push_back(QGauss<dim>(degree + 1));
        quadrature_collection_for_errors.push_back(QGauss<dim>(degree + 2)); // TODO: reduce by one?
      }

    // Add dummy element
    // TODO: Find more elegant solution
    fe_collection_p.push_back(FE_Q<dim, spacedim>(1));
    quadrature_collection_p.push_back(QGauss<dim>(2));
    for (unsigned int degree = 1; degree <= prm.prm_adaptation.max_p_degree - 1; ++degree)
      {
        fe_collection_p.push_back(FE_Q<dim, spacedim>(degree));
        quadrature_collection_p.push_back(QGauss<dim>(degree + 1));
      }

    quadrature_collections = {quadrature_collection_v, quadrature_collection_p};

    // prepare hierarchy
    const unsigned int min_fe_index = prm.prm_adaptation.min_p_degree - 1;

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

    // prepare operators
    a_block_operator =
      std::make_unique<ABlockOperator<dim, LinearAlgebra, spacedim>>(mapping_collection,
                                                                     quadrature_collection_v);

    schur_block_operator =
      std::make_unique<SchurBlockOperator<dim, LinearAlgebra, spacedim>>(mapping_collection,
                                                                         quadrature_collection_p);

    stokes_operator =
      std::make_unique<StokesOperator<dim, LinearAlgebra, spacedim>>(mapping_collection,
                                                                     quadrature_collections);

    // choose functions
    if (prm.grid_type == "kovasznay")
      {
        // boundary_function = Factory::create_function<dim>("zero");
        solution_function_v = Factory::create_function<dim>("kovasznay exact velocity");
        solution_function_p = Factory::create_function<dim>("kovasznay exact pressure");

        rhs_function_v = Factory::create_function<dim>("kovasznay rhs velocity");
        // rhs_function_p = Factory::create_function<dim>("kovasznay rhs pressure");
        rhs_function_p = std::make_shared<Functions::ZeroFunction<dim>>(1);
      }
    else if (prm.grid_type == "y-pipe")
      {
        // boundary_function = Factory::create_function<dim>("zero");

        solution_function_v = Factory::create_function<dim>("zero", /*n_components=*/dim);
        solution_function_p = Factory::create_function<dim>("zero", /*n_components=*/1);

        rhs_function_v = Factory::create_function<dim>("zero", /*n_components=*/dim);
        rhs_function_p = Factory::create_function<dim>("zero", /*n_components=*/1);
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
        // precompute weights per finite element
        std::vector<unsigned int> weights(fe_collection_v.size());
        for (unsigned int i = 0; i < weights.size(); ++i)
          {
            // dofs per cell for all fes with index i
            // unsigned int n_dofs_per_cell = 0;
            // for(const auto &fe_collection : fe_collections)
            //   n_dofs_per_cell += fe_collection[i].n_dofs_per_cell();
            const unsigned int n_dofs_per_cell =
              fe_collection_v[i].n_dofs_per_cell() + fe_collection_p[i].n_dofs_per_cell();

            const float result = prm.prm_adaptation.weighting_factor *
                                 std::pow(n_dofs_per_cell, prm.prm_adaptation.weighting_exponent);

            AssertThrow(result >= 0. &&
                          result <= static_cast<float>(std::numeric_limits<unsigned int>::max()),
                        ExcMessage("Cannot cast determined weight for this cell to unsigned int!"));

            weights[i] = static_cast<unsigned int>(result);
          }

        const auto weighting_function =
          [dof_handler = dof_handlers[0],
           weights](const typename parallel::distributed::Triangulation<dim>::cell_iterator &cell_,
                    const typename parallel::distributed::Triangulation<dim>::CellStatus     status)
          -> unsigned int {
          // get dofs from future fe, and assume all dofhandlers use the same fe index
          const typename DoFHandler<dim, spacedim>::cell_iterator cell(*cell_, dof_handler);

          unsigned int fe_index = numbers::invalid_unsigned_int;
          switch (status)
            {
              case Triangulation<dim, spacedim>::CELL_PERSIST:
              case Triangulation<dim, spacedim>::CELL_REFINE:
              case Triangulation<dim, spacedim>::CELL_INVALID:
                fe_index = cell->future_fe_index();
                break;

              case Triangulation<dim, spacedim>::CELL_COARSEN:
#ifdef DEBUG
                for (const auto &child : cell->child_iterators())
                  Assert(child->is_active() && child->coarsen_flag_set(),
                         typename dealii::Triangulation<dim>::ExcInconsistentCoarseningFlags());
#endif

                fe_index = dealii::internal::hp::DoFHandlerImplementation::
                  dominated_future_fe_on_children<dim, spacedim>(cell);
                break;

              default:
                Assert(false, ExcInternalError());
                break;
            }

          return weights[fe_index];
        };

        weight_connection = triangulation.signals.weight.connect(weighting_function);
      }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  Problem<dim, LinearAlgebra, spacedim>::~Problem()
  {
    weight_connection.disconnect();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::initialize_grid()
  {
    TimerOutput::Scope t(getTimer(), "initialize_grid");

    Factory::create_grid(prm.grid_type, triangulation);

    if (prm.resume_filename.compare("") != 0)
      {
        resume_from_checkpoint();
      }
    else
      {
        // no need to call distribute_dofs here.
        // correct weights are already computed and connected.
        triangulation.refine_global(adaptation_strategy_p->get_n_initial_refinements());

        // set up correct FE indices for hpFull
        const unsigned int min_fe_index = prm.prm_adaptation.min_p_degree - 1;

        for (const auto &cell : dof_handler_v.active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_active_fe_index(min_fe_index);
        for (const auto &cell : dof_handler_p.active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_active_fe_index(min_fe_index);

        dof_handler_v.distribute_dofs(fe_collection_v);
        dof_handler_p.distribute_dofs(fe_collection_p);
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
      partitionings = {&partitioning_v, &partitioning_p};
    }

    {
      TimerOutput::Scope t(getTimer(), "make_constraints");

      {
        constraints_v.clear();
        constraints_v.reinit(partitioning_v.get_owned_dofs(), partitioning_v.get_relevant_dofs());

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

        constraints_v.make_consistent_in_parallel(partitioning_v.get_owned_dofs(),
                                                  partitioning_v.get_active_dofs(),
                                                  mpi_communicator);
        constraints_v.close();
        partitioning_v.get_relevant_dofs() = constraints_v.get_local_lines();
      }

      {
        constraints_p.clear();
        constraints_p.reinit(partitioning_p.get_owned_dofs(), partitioning_p.get_relevant_dofs());

        DoFTools::make_hanging_node_constraints(dof_handler_p, constraints_p);

        constraints_p.make_consistent_in_parallel(partitioning_p.get_owned_dofs(),
                                                  partitioning_p.get_active_dofs(),
                                                  mpi_communicator);
        constraints_p.close();
        partitioning_p.get_relevant_dofs() = constraints_p.get_local_lines();
      }
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
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::solve()
  {
    TimerOutput::Scope t(getTimer(), "solve");

    // We need to introduce a vector that does not contain all ghost elements.
    typename LinearAlgebra::BlockVector completely_distributed_solution;
    stokes_operator->initialize_dof_vector(completely_distributed_solution);

    SolverControl solver_control_refined(system_rhs.size(),
                                         prm.solver_tolerance_factor * system_rhs.l2_norm());

    if (prm.solver_type == "AMG")
      {
        solve_amg<dim, LinearAlgebra, spacedim>(solver_control_refined,
                                                *stokes_operator,
                                                *a_block_operator,
                                                *schur_block_operator,
                                                completely_distributed_solution,
                                                system_rhs);
      }
    else if (prm.solver_type == "GMG")
      {
        const std::string filename_mg_level =
          filename_stem + "-mglevel-cycle_" + std::to_string(cycle) + ".log";

        if (prm.prm_multigrid.smoother_preconditioner_type == "Extended Diagonal")
          {
            solve_gmg<PreconditionExtendedDiagonal<typename LinearAlgebra::Vector>,
                      dim,
                      LinearAlgebra,
                      spacedim>(solver_control_refined,
                                *stokes_operator,
                                *a_block_operator,
                                *schur_block_operator,
                                completely_distributed_solution,
                                system_rhs,
                                prm.prm_multigrid,
                                mapping_collection,
                                quadrature_collection_v,
                                dof_handlers,
                                filename_mg_level);
          }
        else if (prm.prm_multigrid.smoother_preconditioner_type == "ASM")
          {
            solve_gmg<PreconditionASM<typename LinearAlgebra::Vector>,
                      dim,
                      LinearAlgebra,
                      spacedim>(solver_control_refined,
                                *stokes_operator,
                                *a_block_operator,
                                *schur_block_operator,
                                completely_distributed_solution,
                                system_rhs,
                                prm.prm_multigrid,
                                mapping_collection,
                                quadrature_collection_v,
                                dof_handlers,
                                filename_mg_level);
          }
        else if (prm.prm_multigrid.smoother_preconditioner_type == "Diagonal")
          {
            solve_gmg<DiagonalMatrixTimer<typename LinearAlgebra::Vector>,
                      dim,
                      LinearAlgebra,
                      spacedim>(solver_control_refined,
                                *stokes_operator,
                                *a_block_operator,
                                *schur_block_operator,
                                completely_distributed_solution,
                                system_rhs,
                                prm.prm_multigrid,
                                mapping_collection,
                                quadrature_collection_v,
                                dof_handlers,
                                filename_mg_level);
          }
        else
          {
            AssertThrow(false, ExcNotImplemented());
          }
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

    Vector<float> fe_degrees_v(triangulation.n_active_cells());
    for (const auto &cell : dof_handler_v.active_cell_iterators())
      if (cell->is_locally_owned())
        fe_degrees_v(cell->active_cell_index()) = cell->get_fe().degree;

    Vector<float> fe_degrees_p(triangulation.n_active_cells());
    for (const auto &cell : dof_handler_p.active_cell_iterators())
      if (cell->is_locally_owned())
        fe_degrees_p(cell->active_cell_index()) = cell->get_fe().degree;

    Vector<float> subdomain(triangulation.n_active_cells());
    for (auto &subd : subdomain)
      subd = triangulation.locally_owned_subdomain();

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_p);
    data_out.add_data_vector(locally_relevant_solution.block(1), "pressure");
    data_out.add_data_vector(fe_degrees_p, "fe_degree_p");
    data_out.add_data_vector(fe_degrees_v, "fe_degree_v");
    data_out.add_data_vector(subdomain, "subdomain");

    if (adaptation_strategy_p->get_error_estimates().size() > 0)
      data_out.add_data_vector(adaptation_strategy_p->get_error_estimates(), "error");
    if (adaptation_strategy_p->get_hp_indicators().size() > 0)
      data_out.add_data_vector(adaptation_strategy_p->get_hp_indicators(), "hp_indicator");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector(dof_handler_v,
                             locally_relevant_solution.block(0),
                             "velocity",
                             data_component_interpretation);

    data_out.build_patches(mapping_collection);

    data_out.write_vtu_with_pvtu_record("./", prm.file_stem, cycle, mpi_communicator, 2, 1);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::resume_from_checkpoint()
  {
    // extract cycle from filename
    const auto pos = prm.resume_filename.rfind(".cycle-");
    AssertThrow(pos != std::string::npos,
                ExcMessage("Checkpoint: filename misses information about cycle!"));
    const auto substring = prm.resume_filename.substr(pos + 7);
    try
      {
        cycle = std::stoul(substring);
      }
    catch (...)
      {
        AssertThrow(false, ExcMessage("Checkpoint: invalid cycle!"));
      }

    triangulation.load(prm.resume_filename);

    // custom repartitioning using DoFs requires correctly assigned FEs
    dof_handler_p.deserialize_active_fe_indices();
    dof_handler_v.set_active_fe_indices(dof_handler_p.get_active_fe_indices());

    dof_handler_p.distribute_dofs(fe_collection_p);
    dof_handler_v.distribute_dofs(fe_collection_v);

    triangulation.repartition();

    // unpack after repartitioning to avoid unnecessary data transfer
    adaptation_strategy_p->unpack_after_serialization();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::write_to_checkpoint()
  {
    // write triangulation and data
    dof_handler_p.prepare_for_serialization_of_active_fe_indices();
    adaptation_strategy_p->prepare_for_serialization();

    const std::string filename =
      prm.file_stem + ".cycle-" + Utilities::to_string(cycle, 2) + ".checkpoint";
    triangulation.save(filename);

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

          Log::log_cycle(cycle, prm);

          setup_system();

          Log::log_hp_diagnostics(triangulation, dof_handlers, constraints);
          getTable().add_value("dofs_v", dof_handler_v.n_dofs());
          getTable().add_value("dofs_p", dof_handler_p.n_dofs());

// #ifdef DEBUG
          // check if both dofhandlers have same fe indices
          const std::vector<types::fe_index> fe_indices_v = dof_handler_v.get_active_fe_indices();
          const std::vector<types::fe_index> fe_indices_p = dof_handler_p.get_active_fe_indices();
          AssertThrow(std::equal(fe_indices_v.begin(), fe_indices_v.end(), fe_indices_p.begin()),
                      ExcMessage("Active FE indices differ!"));
// #endif

          // ---------- Sanity check limit_p_level_difference ----------
          // ---------- check for ACTIVE fe indices
          for (const auto &cell : dof_handler_v.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
            {
              const auto cell_fe_index = cell->active_fe_index();

              for (unsigned int f = 0; f < cell->n_faces(); ++f)
                if (cell->face(f)->at_boundary() == false)
                  {
                    if (cell->face(f)->has_children())
                      {
                        for (unsigned int sf = 0; sf < cell->face(f)->n_children(); ++sf)
                          {
                            const auto neighbor_subface_fe_index = cell->neighbor_child_on_subface(f, sf)->active_fe_index();

                            if (std::abs(cell_fe_index - neighbor_subface_fe_index) > 1)
                              {
                                std::cout << "cell_fe_index: " << cell_fe_index << std::endl;
                                std::cout << "neighbor_subface_fe_index: " << neighbor_subface_fe_index << std::endl;
                                std::cout << "diff: " << std::abs(cell_fe_index - neighbor_subface_fe_index) << std::endl;
                                AssertThrow(false, ExcMessage("Sanity check fails: subface, stokes."));
                              }
                          }
                      }
                    else
                      {
                        const auto neighbor_fe_index = cell->neighbor(f)->active_fe_index();

                        if (std::abs(cell_fe_index - neighbor_fe_index) > 1)
                          {
                            std::cout << "cell_fe_index: " << cell_fe_index << std::endl;
                            std::cout << "neighbor_fe_index: " << neighbor_fe_index << std::endl;
                            std::cout << "diff: " << std::abs(cell_fe_index - neighbor_fe_index) << std::endl;
                            AssertThrow(false, ExcMessage("Sanity check fails: face, stokes."));
                          }
                      }
                  }

            }
          // --------------------

          a_block_operator->reinit(partitioning_v, dof_handler_v, constraints_v);
          schur_block_operator->reinit(partitioning_p, dof_handler_p, constraints_p);
          stokes_operator->reinit(
            partitionings, dof_handlers, constraints, system_rhs, rhs_functions);

          if (prm.log_nonzero_elements)
            Log::log_nonzero_elements(stokes_operator->get_system_matrix());

          solve();

          if (prm.grid_type == "kovasznay")
            compute_errors();
          adaptation_strategy_p->estimate_mark();

          if ((prm.output_frequency > 0) && (cycle % prm.output_frequency == 0))
            output_results();
        }

        Log::log_timing_statistics(mpi_communicator);

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
