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

#ifndef poisson_problem_h
#define poisson_problem_h


#include <deal.II/distributed/cell_weights.h>
#include <deal.II/distributed/tria.h>

#include <adaptation/base.h>
#include <multigrid/operator_base.h>
#include <parameter.h>
#include <problem_base.h>


namespace Poisson
{
  template <int dim, typename LinearAlgebra, int spacedim = dim>
  class Problem : public ProblemBase
  {
  public:
    Problem(const Parameter &prm);

    void
    run() override;

  private:
    void
    initialize_grid();
    void
    setup_system();
    void
    initialize_system();

    void
    solve();

    void
    compute_errors();
    void
    output_results();

    void
    resume_from_checkpoint();
    void
    write_to_checkpoint();

    MPI_Comm mpi_communicator;

    const Parameter &prm;
    std::string      filename_stem;
    std::string      filename_log;

    dealii::parallel::distributed::Triangulation<dim> triangulation;
    dealii::DoFHandler<dim, spacedim>                 dof_handler;

    dealii::hp::MappingCollection<dim, spacedim> mapping_collection;
    dealii::hp::FECollection<dim, spacedim>      fe_collection;
    dealii::hp::QCollection<dim>                 quadrature_collection;

    std::unique_ptr<Adaptation::Base>            adaptation_strategy;
    dealii::parallel::CellWeights<dim, spacedim> cell_weights;

    std::unique_ptr<dealii::Function<dim>> boundary_function;
    std::unique_ptr<dealii::Function<dim>> solution_function;
    std::unique_ptr<dealii::Function<dim>> rhs_function;

    Partitioning partitioning;

    dealii::AffineConstraints<double> constraints;

    std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>> poisson_operator;

    typename LinearAlgebra::Vector locally_relevant_solution;
    typename LinearAlgebra::Vector system_rhs;

    unsigned int cycle;
  };
} // namespace Poisson


#endif
