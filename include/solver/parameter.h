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
  struct Parameter : public dealii::ParameterAcceptor
  {
    Parameter()
      : dealii::ParameterAcceptor("solver")
    {}
  };
} // namespace Solver


#endif