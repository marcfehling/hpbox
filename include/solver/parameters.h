// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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

#ifndef solver_parameter_h
#define solver_parameter_h


#include <deal.II/base/parameter_acceptor.h>


namespace Solver
{
  struct Parameters : public dealii::ParameterAcceptor
  {
    Parameters()
      : dealii::ParameterAcceptor("solver")
    {
      min_level = 5;
      add_parameter("min level", min_level);

      max_level = 10;
      add_parameter("max level", max_level);

      min_degree = 2;
      add_parameter("min degree", min_degree);

      max_degree = 7;
      add_parameter("max degree", max_degree);

      total_refine_fraction = 0.3;
      add_parameter("total refine fraction", total_refine_fraction);

      total_coarsen_fraction = 0.03;
      add_parameter("total coarsen fraction", total_refine_fraction);

      p_refine_fraction = 0.9;
      add_parameter("p-refine fraction", p_refine_fraction);

      p_coarsen_fraction = 0.9;
      add_parameter("p-coarsen fraction", p_coarsen_fraction);
    }

    unsigned int min_level, max_level;
    unsigned int min_degree, max_degree;

    double total_refine_fraction, total_coarsen_fraction;
    double p_refine_fraction, p_coarsen_fraction;
  };
} // namespace Solver


#endif
