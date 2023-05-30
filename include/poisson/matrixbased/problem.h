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

#ifndef poisson_matrixbased_problem_h
#define poisson_matrixbased_problem_h


// #include <deal.II/distributed/cell_weights.h>
// #include <deal.II/distributed/tria.h>

#include <deal.II/hp/fe_values.h>

// #include <adaptation/base.h>
#include <operator.h>
// #include <parameter.h>
#include <problem.h>
#include <poisson/problem/poisson_operator.h>


namespace Poisson
{
  namespace MatrixBased
  {
    template <int dim, typename LinearAlgebra, int spacedim = dim>
    class Problem : public ProblemBase<dim, LinearAlgebra, spacedim>
    {
    public:
      Problem(const Parameter &prm);

      // void
      // run() override;

    protected: // turn private?
      virtual void
      setup_system() = 0;

      virtual void
      solve_with_amg() = 0;
      virtual void
      solve_with_gmg() = 0;

      std::unique_ptr<dealii::hp::FEValues<dim, spacedim>> fe_values_collection;

      Partitioning partitioning;

      // std::unique_ptr<OperatorType<dim, LinearAlgebra, spacedim>> poisson_operator;
      PoissonMatrixBased::PoissonOperator<dim, LinearAlgebra, spacedim> poisson_operator;
    };
  }
} // namespace Poisson


#endif
