/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */


// TODO: need cleanup
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
  using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/error_predictor.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/refinement.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/smoothness_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <adaptation/factory.h>
#include <adaptation/parameter.h>
#include <function/factory.h>
// #include <operator/poisson/factory.h>
#include <operator/base.h>
#include <operator/poisson/matrix_based.h>
#include <operator/poisson/matrix_free.h>
#include <solver/cg/amg.h>
#include <solver/cg/gmg.h>

#include <fstream>
#include <iostream>
#include <memory>


namespace Poisson
{
  using namespace dealii;



  // @sect3{The <code>Parameter</code> class implementation}

  // Parameter class.

  // forward declarations
  template <int dim>
  class PoissonProblem;



  struct OperatorParameters : public ParameterAcceptor
  {
    OperatorParameters();

    std::string type;

    template <int dim, typename number>
    friend class MGSolverOperatorBase;
    template <int dim>
    friend class PoissonProblem;
  };

  OperatorParameters::OperatorParameters()
    : ParameterAcceptor("operator")
  {
    type = "MatrixFree";
    add_parameter("type", type);
  }



  struct SolverParameters : public ParameterAcceptor
  {
    SolverParameters();

    std::string type;

    template <int dim>
    friend class PoissonProblem;
  };

  SolverParameters::SolverParameters()
    : ParameterAcceptor("solver")
  {
    type = "GMG";
    add_parameter("type", type);
  }



  struct ProblemParameters : public ParameterAcceptor
  {
    ProblemParameters();

    unsigned int dimension;
    unsigned int n_cycles;
    std::string  adaptation_type;

    Adaptation::Parameters prm_adaptation;
    OperatorParameters     prm_operator;
    SolverParameters       prm_solver;

    template <int dim>
    friend class PoissonProblem;
  };

  ProblemParameters::ProblemParameters()
    : ParameterAcceptor("problem")
  {
    dimension = 2;
    add_parameter("dimension", dimension);

    n_cycles = 8;
    add_parameter("n cycles", n_cycles);

    adaptation_type = "hp Legendre";
    add_parameter("adaptation type", adaptation_type);
  }



  // @sect3{Solver and preconditioner}

  // @sect4{Conjugate-gradient solver preconditioned by a algebraic multigrid
  // approach}

  template <typename VectorType, typename OperatorType>
  void
  solve_with_amg(SolverControl &     solver_control,
                 const OperatorType &system_matrix,
                 VectorType &        dst,
                 const VectorType &  src)
  {
    LA::MPI::PreconditionAMG::AdditionalData data;
    data.elliptic              = true;
    data.higher_order_elements = true;

    LA::MPI::PreconditionAMG preconditioner;
    preconditioner.initialize(system_matrix.get_system_matrix(), data);

    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
    cg.solve(system_matrix, dst, src, preconditioner);
  }



  // @sect4{Conjugate-gradient solver preconditioned by hybrid
  // polynomial-global-coarsening multigrid approach}

  template <typename VectorType, typename OperatorType, int dim>
  void
  solve_with_gmg(SolverControl &                  solver_control,
                 const OperatorType &             system_matrix,
                 VectorType &                     dst,
                 const VectorType &               src,
                 const hp::MappingCollection<dim> mapping_collection,
                 const DoFHandler<dim> &          dof_handler,
                 const hp::QCollection<dim> &     quadrature_collection)
  {
    const GMGParameters mg_data; // TODO -> MF

    // Create a DoFHandler and operator for each multigrid level defined
    // by p-coarsening, as well as, create transfer operators.
    MGLevelObject<DoFHandler<dim>> dof_handlers;
    MGLevelObject<std::unique_ptr<
      MGSolverOperatorBase<dim, typename VectorType::value_type>>>
                                                       operators;
    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;

    std::vector<std::shared_ptr<Triangulation<dim>>> coarse_grid_triangulations;
    if (mg_data.perform_h_transfer)
      coarse_grid_triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          dof_handler.get_triangulation());
    else
      coarse_grid_triangulations.emplace_back(
        const_cast<Triangulation<dim> *>(&(dof_handler.get_triangulation())),
        [](auto &) {
          // empty deleter, since fine_triangulation_in is an external field
          // and its destructor is called somewhere else
        });

    const unsigned int n_h_levels = coarse_grid_triangulations.size() - 1;

    // Determine the number of levels.
    const auto get_max_active_fe_index = [&](const auto &dof_handler) {
      unsigned int min = 0;

      for (auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned())
            min = std::max(min, cell->active_fe_index());
        }

      return Utilities::MPI::max(min, MPI_COMM_WORLD);
    };

    const unsigned int n_p_levels =
      MGTransferGlobalCoarseningTools::create_polynomial_coarsening_sequence(
        get_max_active_fe_index(dof_handler) + 1, mg_data.p_sequence)
        .size();

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
                  cell->set_active_fe_index(
                    MGTransferGlobalCoarseningTools::
                      create_next_polynomial_coarsening_degree(
                        cell_other->active_fe_index() + 1, mg_data.p_sequence) -
                    1);
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
        auto &      constraint  = constraints[level];

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

          operators[level] =
            std::make_unique<OperatorType>(mapping_collection,
                                           dof_handler,
                                           quadrature_collection,
                                           constraint,
                                           dummy);
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



  // @sect3{The <code>LaplaceProblem</code> class template}

  // Solving the Laplace equation on subsequently refined function spaces.
  template <int dim>
  class PoissonProblem
  {
  public:
    PoissonProblem(const ProblemParameters &prm);

    void
    run();

  private:
    void
    create_coarse_grid();
    void
    setup_system();

    template <typename Operator>
    void
    solve(const Operator &                            system_matrix,
          LinearAlgebra::distributed::Vector<double> &locally_relevant_solution,
          const LinearAlgebra::distributed::Vector<double> &system_rhs);

    void
    compute_errors();
    void
    output_results(const unsigned int cycle) const;

    MPI_Comm mpi_communicator;

    const ProblemParameters &prm;

    parallel::distributed::Triangulation<dim> triangulation;

    DoFHandler<dim>            dof_handler;
    hp::MappingCollection<dim> mapping_collection;
    hp::FECollection<dim>      fe_collection;
    hp::QCollection<dim>       quadrature_collection;

    std::unique_ptr<hp::FEValues<dim>> fe_values_collection;
    std::unique_ptr<Adaptation::Base>  adaptation_strategy;

    std::unique_ptr<Function<dim>> boundary_function;
    std::unique_ptr<Function<dim>> solution_function;
    std::unique_ptr<Function<dim>> rhs_function;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    LA::MPI::SparseMatrix                      system_matrix;
    LinearAlgebra::distributed::Vector<double> locally_relevant_solution;
    LinearAlgebra::distributed::Vector<double> system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };



  template <int dim>
  PoissonProblem<dim>::PoissonProblem(const ProblemParameters &prm)
    : mpi_communicator(MPI_COMM_WORLD)
    , prm(prm)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {
    TimerOutput::Scope t(computing_timer, "init");

    // prepare collections
    mapping_collection.push_back(MappingQ1<dim>());

    const unsigned int min_degree = prm.prm_adaptation.min_degree,
                       max_degree = prm.prm_adaptation.max_degree;
    for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
      {
        fe_collection.push_back(FE_Q<dim>(degree));
        quadrature_collection.push_back(QGauss<dim>(degree + 1));
      }

    // prepare fe values
    fe_values_collection =
      std::make_unique<hp::FEValues<dim>>(fe_collection,
                                          quadrature_collection,
                                          update_values | update_gradients |
                                            update_quadrature_points |
                                            update_JxW_values);
    fe_values_collection->precalculate_fe_values();

    // choose functions
    boundary_function = Factory::create_function<dim>("reentrant corner");
    solution_function = Factory::create_function<dim>("reentrant corner");
    rhs_function      = Factory::create_function<dim>("zero");

    // choose adaptation strategy
    adaptation_strategy =
      Factory::create_adaptation<dim>(prm.adaptation_type,
                                      prm.prm_adaptation,
                                      locally_relevant_solution,
                                      fe_collection,
                                      dof_handler,
                                      triangulation);
  }



  template <int dim>
  void
  PoissonProblem<dim>::create_coarse_grid()
  {
    TimerOutput::Scope t(computing_timer, "coarse grid");

    std::vector<unsigned int> repetitions(dim, 2);
    Point<dim>                bottom_left, top_right;
    for (unsigned int d = 0; d < dim; ++d)
      {
        bottom_left[d] = -1.;
        top_right[d]   = 1.;
      }

    std::vector<int> cells_to_remove(dim, 1);
    cells_to_remove[0] = -1;

    // TODO
    // expand domain by 1 cell in z direction for 3d case

    GridGenerator::subdivided_hyper_L(
      triangulation, repetitions, bottom_left, top_right, cells_to_remove);
  }



  template <int dim>
  void
  PoissonProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

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
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             *boundary_function,
                                             constraints);

#ifdef DEBUG
    // We have not dealt with chains of constraints on ghost cells yet.
    // Thus, we are content with verifying their consistency for now.
    IndexSet locally_active_dofs;
    DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);
    AssertThrow(constraints.is_consistent_in_parallel(
                  Utilities::MPI::all_gather(mpi_communicator,
                                             dof_handler.locally_owned_dofs()),
                  locally_active_dofs,
                  mpi_communicator,
                  /*verbose=*/true),
                ExcMessage(
                  "AffineConstraints object contains inconsistencies!"));
#endif
    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
  }



  template <int dim>
  template <typename OperatorType>
  void
  PoissonProblem<dim>::solve(
    const OperatorType &                              system_matrix,
    LinearAlgebra::distributed::Vector<double> &      locally_relevant_solution,
    const LinearAlgebra::distributed::Vector<double> &system_rhs)
  {
    TimerOutput::Scope t(computing_timer, "solve");

    LinearAlgebra::distributed::Vector<double> locally_relevant_solution_;
    LinearAlgebra::distributed::Vector<double> system_rhs_;

    system_matrix.initialize_dof_vector(locally_relevant_solution_);
    system_matrix.initialize_dof_vector(system_rhs_);

    system_rhs_.copy_locally_owned_data_from(system_rhs);

    SolverControl solver_control(system_rhs_.size(),
                                 1e-12 * system_rhs_.l2_norm());

    if (prm.prm_solver.type == "AMG")
      solve_with_amg(solver_control,
                     system_matrix,
                     locally_relevant_solution_,
                     system_rhs_);
    else if (prm.prm_solver.type == "GMG")
      solve_with_gmg(solver_control,
                     system_matrix,
                     locally_relevant_solution_,
                     system_rhs_,
                     mapping_collection,
                     dof_handler,
                     quadrature_collection);
    else
      Assert(false, ExcNotImplemented());

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(locally_relevant_solution_);

    locally_relevant_solution.copy_locally_owned_data_from(
      locally_relevant_solution_);
    locally_relevant_solution.update_ghost_values();
  }



  template <int dim>
  void
  PoissonProblem<dim>::compute_errors()
  {
    TimerOutput::Scope t(computing_timer, "compute errors");

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

    pcout << "L2 error: " << L2_error << std::endl
          << "H1 error: " << H1_error << std::endl;

    // TODO
    // Store errors in Convergence table
  }



  template <int dim>
  void
  PoissonProblem<dim>::output_results(const unsigned int cycle) const
  {
    Vector<float> fe_degrees(triangulation.n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        fe_degrees(cell->active_cell_index()) =
          fe_collection[cell->active_fe_index()].degree;

    Vector<float> subdomain(triangulation.n_active_cells());
    for (auto &subd : subdomain)
      subd = triangulation.locally_owned_subdomain();

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "solution");
    data_out.add_data_vector(fe_degrees, "fe_degree");
    data_out.add_data_vector(subdomain, "subdomain");
    // data_out.add_data_vector(estimated_error_per_cell, "error");
    // data_out.add_data_vector(hp_decision_indicators, "hp_indicator");
    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2, 8);
  }



  template <int dim>
  void
  PoissonProblem<dim>::run()
  {
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    // create operator
    // TODO: I am not happy with this
    std::unique_ptr<Operator::Poisson::MatrixBased<
      dim,
      LinearAlgebra::distributed::Vector<double>>>
      poisson_operator_matrixbased;
    std::unique_ptr<
      Operator::Poisson::MatrixFree<dim,
                                    LinearAlgebra::distributed::Vector<double>>>
      poisson_operator_matrixfree;
    if (prm.prm_operator.type == "MatrixBased")
      poisson_operator_matrixbased = std::make_unique<
        Operator::Poisson::
          MatrixBased<dim, LinearAlgebra::distributed::Vector<double>>>();
    else if (prm.prm_operator.type == "MatrixFree")
      poisson_operator_matrixfree = std::make_unique<
        Operator::Poisson::
          MatrixFree<dim, LinearAlgebra::distributed::Vector<double>>>();
    else
      Assert(false, ExcNotImplemented());

    for (unsigned int cycle = 0;
         cycle < (prm.adaptation_type != "hp history" ? prm.n_cycles :
                                                        prm.n_cycles + 1);
         ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            create_coarse_grid();

            const unsigned int min_level = prm.prm_adaptation.min_level;
            triangulation.refine_global(
              prm.adaptation_type != "hp history" ? min_level : min_level - 1);
          }
        else
          {
            adaptation_strategy->estimate_mark_refine();
          }

        setup_system();

        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

        // TODO: I am not happy with this
        if (poisson_operator_matrixbased != nullptr)
          {
            poisson_operator_matrixbased->reinit(mapping_collection,
                                                 dof_handler,
                                                 quadrature_collection,
                                                 constraints,
                                                 system_rhs);
            solve(*poisson_operator_matrixbased,
                  locally_relevant_solution,
                  system_rhs);
          }
        else if (poisson_operator_matrixfree != nullptr)
          {
            std::cout << "we are here" << std::endl;
            poisson_operator_matrixfree->reinit(mapping_collection,
                                                dof_handler,
                                                quadrature_collection,
                                                constraints,
                                                system_rhs);
            solve(*poisson_operator_matrixfree,
                  locally_relevant_solution,
                  system_rhs);
          }
        else
          Assert(false, ExcInternalError());

        compute_errors();

        if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
          {
            TimerOutput::Scope t(computing_timer, "output");
            output_results(cycle);
          }

        computing_timer.print_summary();
        computing_timer.reset();

        pcout << std::endl;
      }
  }
} // namespace Poisson



int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Poisson;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      ProblemParameters prm_problem;

      const std::string filename        = (argc > 1) ? argv[1] : "",
                        output_filename = (argc > 1) ? "" : "poisson.prm";
      ParameterAcceptor::initialize(filename, output_filename);

      const int dim = prm_problem.dimension;
      if (dim == 2)
        {
          PoissonProblem<2> poisson_problem(prm_problem);
          poisson_problem.run();
        }
      else if (dim == 3)
        {
          PoissonProblem<3> poisson_problem(prm_problem);
          poisson_problem.run();
        }
      else
        Assert(false, ExcNotImplemented());
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
