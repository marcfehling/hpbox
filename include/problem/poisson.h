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

#ifndef problem_poisson_h
#define problem_poisson_h


#include <deal.II/distributed/tria.h>

#include <deal.II/hp/fe_values.h>

#include <adaptation/base.h>
#include <operator/poisson/matrix_based.h>
#include <operator/poisson/matrix_free.h>
#include <problem/base.h>
#include <problem/parameter.h>


namespace Problem
{
  template <int dim, typename LinearAlgebra, int spacedim = dim>
  class Poisson : public Base
  {
  public:
    Poisson(const Parameters &prm);

    void
    run() override;

  private:
    void
    initialize_grid();
    void
    setup_system();
    void
    log_diagnostics();

    template <typename OperatorType>
    void
    solve(const OperatorType &                  system_matrix,
          typename LinearAlgebra::Vector &      locally_relevant_solution,
          const typename LinearAlgebra::Vector &system_rhs);

    void
    compute_errors();
    void
    output_results(const unsigned int cycle);
    void
    log_timings();

    MPI_Comm mpi_communicator;

    const Parameters &prm;
    std::string       filename_log;

    dealii::parallel::distributed::Triangulation<dim> triangulation;
    dealii::DoFHandler<dim, spacedim>                 dof_handler;

    dealii::hp::MappingCollection<dim, spacedim> mapping_collection;
    dealii::hp::FECollection<dim, spacedim>      fe_collection;
    dealii::hp::QCollection<dim>                 quadrature_collection;

    std::unique_ptr<dealii::hp::FEValues<dim, spacedim>> fe_values_collection;
    std::unique_ptr<Adaptation::Base>                    adaptation_strategy;

    std::unique_ptr<dealii::Function<dim>> boundary_function;
    std::unique_ptr<dealii::Function<dim>> solution_function;
    std::unique_ptr<dealii::Function<dim>> rhs_function;

    dealii::IndexSet locally_owned_dofs;
    dealii::IndexSet locally_relevant_dofs;

    dealii::AffineConstraints<double> constraints;

    // TODO: Base class?
    std::unique_ptr<
      Operator::Poisson::MatrixBased<dim, LinearAlgebra, spacedim>>
      poisson_operator_matrixbased;

    // only exists for distributed vector
    std::unique_ptr<Operator::Poisson::MatrixFree<dim, LinearAlgebra, spacedim>>
      poisson_operator_matrixfree;

    typename LinearAlgebra::Vector locally_relevant_solution;
    typename LinearAlgebra::Vector system_rhs;
  };
} // namespace Problem


#endif
