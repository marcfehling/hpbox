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

    // virtual void
    // run() override;

  protected:
    virtual void
    initialize_grid();

//    virtual void
//    setup_system() = 0;

//    virtual void
//    solve() = 0;

    virtual void
    compute_errors();
    virtual void
    output_results();

    virtual void
    resume_from_checkpoint();
    virtual void
    write_to_checkpoint();

    MPI_Comm mpi_communicator;

    const Parameter &prm;

    dealii::parallel::distributed::Triangulation<dim, spacedim> triangulation;
    dealii::DoFHandler<dim, spacedim>                           dof_handler;

    dealii::hp::MappingCollection<dim, spacedim> mapping_collection;
    dealii::hp::FECollection<dim, spacedim>      fe_collection;
    dealii::hp::QCollection<dim>                 quadrature_collection;

    std::unique_ptr<Adaptation::Base> adaptation_strategy;

    std::unique_ptr<dealii::Function<dim>> solution_function;

    typename LinearAlgebra::Vector locally_relevant_solution;

    unsigned int cycle;
};


#endif
