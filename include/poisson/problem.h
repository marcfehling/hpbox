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

#ifndef poisson_poisson_h
#define poisson_poisson_h


#include <deal.II/distributed/tria.h>

#include <deal.II/hp/fe_values.h>

#include <adaptation/base.h>
#include <parameter.h>
#include <poisson/matrixbased.h>
#include <poisson/matrixfree.h>
#include <problem.h>


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

    template <typename OperatorType>
    void
    solve(const OperatorType                   &system_matrix,
          typename LinearAlgebra::Vector       &locally_relevant_solution,
          const typename LinearAlgebra::Vector &system_rhs);

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
    std::string      filename_log;

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
    std::unique_ptr<OperatorMatrixBased<dim, LinearAlgebra, spacedim>>
      operator_matrixbased;

    // only exists for distributed vector
    std::unique_ptr<OperatorMatrixFree<dim, LinearAlgebra, spacedim>>
      operator_matrixfree;

    typename LinearAlgebra::Vector locally_relevant_solution;
    typename LinearAlgebra::Vector system_rhs;

    unsigned int cycle;
  };
} // namespace Poisson


#endif
