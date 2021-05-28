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

#ifndef adaptation_parameter_h
#define adaptation_parameter_h


#include <deal.II/base/parameter_acceptor.h>


namespace Adaptation
{
  struct Parameters : public dealii::ParameterAcceptor
  {
    Parameters()
      : dealii::ParameterAcceptor("adaptation")
    {
      min_h_level = 5;
      add_parameter("min level", min_h_level);

      max_h_level = 10;
      add_parameter("max level", max_h_level);

      min_p_degree = 2;
      add_parameter("min degree", min_p_degree);

      max_p_degree = 7;
      add_parameter("max degree", max_p_degree);

      total_refine_fraction = 0.3;
      add_parameter("total refine fraction", total_refine_fraction);

      total_coarsen_fraction = 0.03;
      add_parameter("total coarsen fraction", total_refine_fraction);

      p_refine_fraction = 0.9;
      add_parameter("p-refine fraction", p_refine_fraction);

      p_coarsen_fraction = 0.9;
      add_parameter("p-coarsen fraction", p_coarsen_fraction);

      weighting_factor = 1e6;
      add_parameter("weighting factor", weighting_factor);

      weighting_exponent = 1.;
      add_parameter("weighting exponent", weighting_exponent);
    }

    unsigned int min_h_level, max_h_level;
    unsigned int min_p_degree, max_p_degree;

    double total_refine_fraction, total_coarsen_fraction;
    double p_refine_fraction, p_coarsen_fraction;

    double weighting_factor, weighting_exponent;
  };
} // namespace Adaptation


#endif
