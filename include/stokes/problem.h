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

#ifndef stokes_problem_h
#define stokes_problem_h


#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/hp/fe_values.h>

#include <adaptation/base.h>
#include <parameter.h>
#include <problem.h>


namespace Stokes
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

    // TODO: we go with the classical matrix based way for now
    // template <typename OperatorType>
    // void
    // solve(const OperatorType                        &system_matrix,
    //       typename LinearAlgebra::BlockVector &locally_relevant_solution,
    //       const typename LinearAlgebra::BlockVector &system_rhs);
    void
    assemble_system();
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
    std::string      filename_log;

    dealii::parallel::distributed::Triangulation<dim> triangulation;
    dealii::DoFHandler<dim, spacedim>                 dof_handler;

    // or in operator class, or in both?
    const dealii::FEValuesExtractors::Vector velocities;
    const dealii::FEValuesExtractors::Scalar pressure;

    dealii::hp::MappingCollection<dim, spacedim> mapping_collection;
    dealii::hp::FECollection<dim, spacedim>      fe_collection;
    dealii::hp::QCollection<dim>                 quadrature_collection;
    dealii::hp::QCollection<dim>                 quadrature_collection_for_errors;

    std::unique_ptr<dealii::hp::FEValues<dim, spacedim>> fe_values_collection;
    std::unique_ptr<Adaptation::Base>                    adaptation_strategy;

    std::unique_ptr<dealii::Function<dim>> boundary_function;
    std::unique_ptr<dealii::Function<dim>> solution_function;
    std::unique_ptr<dealii::Function<dim>> rhs_function;

    Partitioning partitioning;

    dealii::AffineConstraints<double> constraints;

    // TODO: stick to classical matrix based appraoch
    // std::unique_ptr<
    //  Operator::Stokes::MatrixBased<dim, LinearAlgebra, spacedim>>
    //  stokes_operator_matrixbased;
    typename LinearAlgebra::BlockSparseMatrix system_matrix;
    typename LinearAlgebra::BlockSparseMatrix preconditioner_matrix;

    typename LinearAlgebra::BlockVector locally_relevant_solution;
    typename LinearAlgebra::BlockVector system_rhs;

    unsigned int cycle;
  };
} // namespace Stokes


#endif
