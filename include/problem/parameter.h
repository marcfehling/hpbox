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

#ifndef problem_parameter_h
#define problem_parameter_h


#include <deal.II/base/parameter_acceptor.h>

#include <adaptation/parameter.h>


namespace Problem
{
  struct Parameters : public dealii::ParameterAcceptor
  {
    Parameters()
      : dealii::ParameterAcceptor("problem")
    {
      dimension = 2;
      add_parameter("dimension", dimension);

      adaptation_type = "hp Legendre";
      add_parameter("adaptation type", adaptation_type);

      operator_type = "MatrixFree";
      add_parameter("operator type", operator_type);

      problem_type = "Poisson";
      add_parameter("problem type", problem_type);

      solver_type = "GMG";
      add_parameter("solver type", solver_type);
    }

    unsigned int dimension;

    std::string adaptation_type;
    std::string operator_type;
    std::string solver_type;
    std::string problem_type;

    Adaptation::Parameters prm_adaptation;
  };
} // namespace Problem


#endif
