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

#ifndef stokes_matrixfree_problem_h
#define stokes_matrixfree_problem_h


#include <deal.II/distributed/cell_weights.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/hp/fe_values.h>

#include <adaptation/base.h>
#include <parameter.h>
#include <problem.h>
#include <stokes/matrixfree/stokes_operator.h>


namespace StokesMatrixFree
{
  template <int dim, typename LinearAlgebra, int spacedim = dim>
  class Problem : public ProblemInterface
  {
  public:
    Problem(const Parameter &prm);

    ~Problem();

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

    dealii::DoFHandler<dim, spacedim>                      dof_handler_v;
    dealii::DoFHandler<dim, spacedim>                      dof_handler_p;
    std::vector<const dealii::DoFHandler<dim, spacedim> *> dof_handlers;

    dealii::hp::MappingCollection<dim, spacedim> mapping_collection;

    dealii::hp::FECollection<dim, spacedim> fe_collection_v;
    dealii::hp::FECollection<dim, spacedim> fe_collection_p;

    dealii::hp::QCollection<dim>              quadrature_collection_v;
    dealii::hp::QCollection<dim>              quadrature_collection_p;
    std::vector<dealii::hp::QCollection<dim>> quadrature_collections;
    dealii::hp::QCollection<dim>              quadrature_collection_for_errors;

    std::unique_ptr<dealii::hp::FEValues<dim, spacedim>> fe_values_collection;
    std::unique_ptr<Adaptation::Base>                    adaptation_strategy_p;

    boost::signals2::connection weight_connection;

    std::unique_ptr<dealii::Function<spacedim>> boundary_function_v;
    std::unique_ptr<dealii::Function<spacedim>> boundary_function_p;
    std::unique_ptr<dealii::Function<spacedim>> solution_function_v;
    std::unique_ptr<dealii::Function<spacedim>> solution_function_p;

    std::shared_ptr<dealii::Function<spacedim>>     rhs_function_v;
    std::shared_ptr<dealii::Function<spacedim>>     rhs_function_p;
    std::vector<const dealii::Function<spacedim> *> rhs_functions;

    Partitioning                      partitioning_v;
    Partitioning                      partitioning_p;
    std::vector<const Partitioning *> partitionings;

    dealii::AffineConstraints<double>                      constraints_v;
    dealii::AffineConstraints<double>                      constraints_p;
    std::vector<const dealii::AffineConstraints<double> *> constraints;

    std::unique_ptr<StokesMatrixFree::StokesOperator<dim, LinearAlgebra, spacedim>> stokes_operator;
    std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>> a_block_operator;
    std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>> schur_block_operator;

    typename LinearAlgebra::BlockVector locally_relevant_solution;
    typename LinearAlgebra::BlockVector system_rhs;

    unsigned int cycle;
  };
} // namespace StokesMatrixFree


#endif
