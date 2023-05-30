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

#ifndef problem_h
#define problem_h

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <parameter.h>


class ProblemInterface
{
public:
  virtual ~ProblemInterface() = default;

  virtual void
  run() = 0;
};


template<int dim, typename LinearAlgebra, int spacedim = dim>
class ProblemBase : public ProblemInterface
{
  public:
    ProblemBase() = default;
    virtual ~ProblemBase() = default;

    virtual void
    run() override;

  protected:
    virtual void
    setup_system() = 0;

    virtual void
    solve_with_amg(LinearAlgebra::Vector &completely_distributed_solution);
    virtual void
    solve_with_gmg(LinearAlgebra::Vector &completely_distributed_solution);

// TODO: turn these functions private
    void
    initialize_grid();

    void
    compute_errors();
    void
    output_results();

    void
    solve();

    void
    resume_from_checkpoint();
    void
    write_to_checkpoint();

    MPI_Comm mpi_communicator;

    const Parameter &prm;
    std::string      filename_log;

    dealii::parallel::distributed::Triangulation<dim, spacedim> triangulation;
    dealii::DoFHandler<dim, spacedim>                           dof_handler;

    dealii::hp::MappingCollection<dim, spacedim> mapping_collection;
    dealii::hp::FECollection<dim, spacedim>      fe_collection;
    dealii::hp::QCollection<dim>                 quadrature_collection;

    std::unique_ptr<Adaptation::Base>            adaptation_strategy;
    dealii::parallel::CellWeights<dim, spacedim> cell_weights;

    dealii::AffineConstraints<double> constraints;

    std::unique_ptr<dealii::Function<dim>> boundary_function;
    std::unique_ptr<dealii::Function<dim>> rhs_function;
    std::unique_ptr<dealii::Function<dim>> solution_function;

    typename LinearAlgebra::Vector locally_relevant_solution;
    typename LinearAlgebra::Vector system_rhs;

    unsigned int cycle;


    // also introduce component masks for Stokes
};


#endif
