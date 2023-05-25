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

#ifndef poisson_problem_h
#define poisson_problem_h


#include <deal.II/distributed/cell_weights.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/hp/fe_values.h>

#include <adaptation/base.h>
#include <operator.h>
#include <parameter.h>
#include <problem.h>


namespace Poisson
{
  template <int dim, typename LinearAlgebra, int spacedim = dim>
  class Problem : public ProblemBase<dim, LinearAlgebra, spacedim>
  {
  public:
    Problem(const Parameter &prm);

    void
    run() override;

  private:
//    void
//    initialize_grid();
    void
    setup_system();
    void
    initialize_system();

    void
    solve();

//    void
//    compute_errors();
//    void
//    output_results();

//    void
//    resume_from_checkpoint();
//    void
//    write_to_checkpoint();

    std::string      filename_log;

    std::unique_ptr<dealii::hp::FEValues<dim, spacedim>> fe_values_collection;
    dealii::parallel::CellWeights<dim, spacedim>         cell_weights;

    std::unique_ptr<dealii::Function<dim>> boundary_function;
    std::unique_ptr<dealii::Function<dim>> rhs_function;

    Partitioning partitioning;

    dealii::AffineConstraints<double> constraints;

    std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>> poisson_operator;

    typename LinearAlgebra::Vector system_rhs;
  };
} // namespace Poisson


#endif
