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

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <base/global.h>
#include <base/linear_algebra.h>
#include <base/log.h>
#include <factory.h>
#include <stokes/problem.h>

//#include <ctime>
//#include <iomanip>
//#include <sstream>

using namespace dealii;


namespace LinearSolvers
{
  template <typename LinearAlgebra>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(const typename LinearAlgebra::BlockSparseMatrix  &S,
                             const typename LinearAlgebra::BlockSparseMatrix  &Spre,
                             const typename LinearAlgebra::PreconditionJacobi &Mppreconditioner,
                             const typename LinearAlgebra::PreconditionAMG    &Apreconditioner,
                             const bool                                        do_solve_A)
      : stokes_matrix(&S)
      , stokes_preconditioner_matrix(&Spre)
      , mp_preconditioner(Mppreconditioner)
      , a_preconditioner(Apreconditioner)
      , do_solve_A(do_solve_A)
    {}

    void
    vmult(typename LinearAlgebra::BlockVector       &dst,
          const typename LinearAlgebra::BlockVector &src) const
    {
      typename LinearAlgebra::Vector utmp(src.block(0));

      {
        SolverControl solver_control(5000, 1e-6 * src.block(1).l2_norm());

        SolverCG<typename LinearAlgebra::Vector> solver(solver_control);

        solver.solve(stokes_preconditioner_matrix->block(1, 1),
                     dst.block(1),
                     src.block(1),
                     mp_preconditioner);

        dst.block(1) *= -1.0;
      }

      {
        stokes_matrix->block(0, 1).vmult(utmp, dst.block(1));
        utmp *= -1.0;
        utmp += src.block(0);
      }

      if (do_solve_A == true)
        {
          SolverControl                    solver_control(5000, utmp.l2_norm() * 1e-2);
          typename LinearAlgebra::SolverCG solver(solver_control);
          solver.solve(stokes_matrix->block(0, 0), dst.block(0), utmp, a_preconditioner);
        }
      else
        a_preconditioner.vmult(dst.block(0), utmp);
    }

  private:
    const SmartPointer<const typename LinearAlgebra::BlockSparseMatrix> stokes_matrix;
    const SmartPointer<const typename LinearAlgebra::BlockSparseMatrix>
                                                      stokes_preconditioner_matrix;
    const typename LinearAlgebra::PreconditionJacobi &mp_preconditioner;
    const typename LinearAlgebra::PreconditionAMG    &a_preconditioner;
    const bool                                        do_solve_A;
  };
} // namespace LinearSolvers



namespace Stokes
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
    }

    partitioning.reinit(dof_handler, stokes_sub_blocks);

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
  Problem<dim, LinearAlgebra, spacedim>::initialize_system()
  {
    TimerOutput::Scope t(getTimer(), "initialize_system");

    {
      TimerOutput::Scope t(getTimer(), "reinit_matrices");

      system_matrix.clear();

      Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (!((c == dim) && (d == dim)))
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      initialize_block_sparse_matrix(
        system_matrix, dof_handler, constraints, partitioning, coupling);

      preconditioner_matrix.clear();

      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (c == d)
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      initialize_block_sparse_matrix(
        preconditioner_matrix, dof_handler, constraints, partitioning, coupling);
    }

    {
      TimerOutput::Scope(getTimer(), "reinit_vectors");

      locally_relevant_solution.reinit(partitioning.get_owned_dofs_per_block(),
                                       partitioning.get_relevant_dofs_per_block(),
                                       mpi_communicator);
      system_rhs.reinit(partitioning.get_owned_dofs_per_block(), mpi_communicator);
    }

    if (prm.operator_type == "MatrixBased")
      Log::log_nonzero_elements(system_matrix);
  }



  // TODO: we stick to the classical matrix based appraoch for now
  /*
  template <int dim, typename LinearAlgebra, int spacedim>
  template <typename OperatorType>
  void
  Problem<dim, LinearAlgebra, spacedim>::solve(
    const OperatorType                        &system_matrix,
    typename LinearAlgebra::BlockVector       &locally_relevant_solution,
    const typename LinearAlgebra::BlockVector &system_rhs)
  {
    Assert(false, ExcNotImplemented());

    TimerOutput::Scope t(getTimer(), "solve");

    // Note: Only change to 'Poisson' is use of BlockVectors (and reinit below)
    typename LinearAlgebra::BlockVector completely_distributed_solution;
    typename LinearAlgebra::BlockVector completely_distributed_system_rhs;

    // TODO: Maybe use initialization functions and overloads for each vector
  type if constexpr (std::is_same< typename LinearAlgebra::Vector,
                    dealii::LinearAlgebra::distributed::Vector<double>>::value)
      {
        Assert(false, ExcNotImplemented());
      }
    else
      {
        completely_distributed_solution.reinit(owned_partitioning,
                                               mpi_communicator);
        completely_distributed_system_rhs = system_rhs;
      }

    SolverControl solver_control(completely_distributed_system_rhs.size(),
                                 1e-12 *
                                   completely_distributed_system_rhs.l2_norm());



    getPCOut() << "   Number of iterations:         "
               << solver_control.last_step() << std::endl;
    getTable().add_value("iteratations", solver_control.last_step());

    constraints.distribute(completely_distributed_solution);

    if constexpr (std::is_same<
                    typename LinearAlgebra::Vector,
                    dealii::LinearAlgebra::distributed::Vector<double>>::value)
      {
        Assert(false, ExcNotImplemented());
      }
    else
      {
        locally_relevant_solution = completely_distributed_solution;
      }
  }
  */



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::assemble_system()
  {
    TimerOutput::Scope t(getTimer(), "assemble_system");

    system_matrix         = 0;
    preconditioner_matrix = 0;
    system_rhs            = 0;

    FullMatrix<double> cell_matrix;
    FullMatrix<double> cell_matrix2;
    Vector<double>     cell_rhs;

    std::vector<Vector<double>> rhs_values;

    std::vector<Tensor<2, dim>> grad_phi_u;
    std::vector<double>         div_phi_u;
    std::vector<double>         phi_p;

    std::vector<types::global_dof_index> local_dof_indices;
    const FEValuesExtractors::Vector     velocities(0);
    const FEValuesExtractors::Scalar     pressure(dim);
    for (const auto &cell :
         dof_handler.active_cell_iterators() | IteratorFilters::LocallyOwnedCell())
      {
        fe_values_collection->reinit(cell);

        const FEValues<dim> &fe_values     = fe_values_collection->get_present_fe_values();
        const unsigned int   n_q_points    = fe_values.n_quadrature_points;
        const unsigned int   dofs_per_cell = fe_values.dofs_per_cell;

        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_matrix = 0;
        cell_matrix2.reinit(dofs_per_cell, dofs_per_cell);
        cell_matrix2 = 0;
        cell_rhs.reinit(dofs_per_cell);
        cell_rhs = 0;

        grad_phi_u.resize(dofs_per_cell);
        div_phi_u.resize(dofs_per_cell);
        phi_p.resize(dofs_per_cell);

        local_dof_indices.resize(dofs_per_cell);

        // TODO: Move this part to the problem class???
        //       Not possible...
        rhs_values.resize(n_q_points, Vector<double>(dim + 1));
        rhs_function->vector_value_list(fe_values.get_quadrature_points(), rhs_values);

        // TODO: move to parameter
        const double viscosity = 0.1;

        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points; ++q_point)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                grad_phi_u[k] = fe_values[velocities].gradient(k, q_point);
                div_phi_u[k]  = fe_values[velocities].divergence(k, q_point);
                phi_p[k]      = fe_values[pressure].value(k, q_point);
              }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    const double tmp = viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]);

                    cell_matrix(i, j) += (tmp - div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
                                         fe_values.JxW(q_point);

                    cell_matrix2(i, j) +=
                      (tmp + 1.0 / viscosity * phi_p[i] * phi_p[j]) * fe_values.JxW(q_point);
                  }

                const unsigned int component_i = cell->get_fe().system_to_component_index(i).first;
                cell_rhs(i) += fe_values.shape_value(i, q_point) *
                               rhs_values[q_point](component_i) * fe_values.JxW(q_point);
              }
          }
        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);

        constraints.distribute_local_to_global(cell_matrix2,
                                               local_dof_indices,
                                               preconditioner_matrix);
      }

    system_rhs.compress(VectorOperation::values::add);
    system_matrix.compress(VectorOperation::values::add);
    preconditioner_matrix.compress(VectorOperation::add);
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  Problem<dim, LinearAlgebra, spacedim>::solve()
  {
    TimerOutput::Scope t(getTimer(), "solve");

    typename LinearAlgebra::PreconditionJacobi Mp_preconditioner;
    typename LinearAlgebra::PreconditionAMG    Amg_preconditioner;

    typename LinearAlgebra::PreconditionAMG::AdditionalData Amg_data;
    if constexpr (std::is_same<LinearAlgebra, PETSc>::value)
      {
        Amg_data.symmetric_operator = true;
        Amg_data.n_sweeps_coarse    = 2;
        Amg_data.strong_threshold   = 0.02;
      }
    else if constexpr (std::is_same<LinearAlgebra, Trilinos>::value ||
                       std::is_same<LinearAlgebra, dealiiTrilinos>::value)
      {
        std::vector<std::vector<bool>> constant_modes;
        DoFTools::extract_constant_modes(dof_handler,
                                         fe_collection.component_mask(velocities),
                                         constant_modes);

        Amg_data.constant_modes        = constant_modes;
        Amg_data.elliptic              = true;
        Amg_data.higher_order_elements = true;
        Amg_data.smoother_sweeps       = 2;
        Amg_data.aggregation_threshold = 0.02;
      }
    else
      {
        Assert(false, dealii::ExcNotImplemented());
      }


    Mp_preconditioner.initialize(preconditioner_matrix.block(1, 1));
    Amg_preconditioner.initialize(preconditioner_matrix.block(0, 0), Amg_data);



    typename LinearAlgebra::BlockVector completely_distributed_solution(
      partitioning.get_owned_dofs_per_block(), mpi_communicator);
    constraints.set_zero(completely_distributed_solution);


    {
      const LinearSolvers::BlockSchurPreconditioner<LinearAlgebra> preconditioner(
        system_matrix, preconditioner_matrix, Mp_preconditioner, Amg_preconditioner, true);

      SolverControl solver_control_refined(system_matrix.m(), 1e-8 * system_rhs.l2_norm());

      PrimitiveVectorMemory<typename LinearAlgebra::BlockVector> mem;

      typename SolverFGMRES<typename LinearAlgebra::BlockVector>::AdditionalData fgmres_data(50);
      SolverFGMRES<typename LinearAlgebra::BlockVector> solver(solver_control_refined,
                                                               mem,
                                                               fgmres_data);

      solver.solve(system_matrix, completely_distributed_solution, system_rhs, preconditioner);

      Log::log_iterations(solver_control_refined);
    }


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
          initialize_system();

          // TODO: I am not happy with this
          if (prm.operator_type == "MatrixBased")
            {
              assemble_system();
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
