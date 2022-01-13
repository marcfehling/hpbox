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

#ifndef problem_factory_h
#define problem_factory_h


#include <deal.II/base/exceptions.h>

#include <problem/poisson.h>
#include <problem/stokes.h>


namespace Factory
{
  template <int dim,
            typename LinearAlgebra,
            int spacedim = dim,
            typename... Args>
  std::unique_ptr<Problem::Base>
  create_problem(const std::string &type, Args &&...args)
  {
    if (type == "poisson")
      return std::make_unique<Problem::Poisson<dim, LinearAlgebra, spacedim>>(
        std::forward<Args>(args)...);
    /*
    else if (type == "stokes")
      return std::make_unique<Problem::Stokes<dim, LinearAlgebra, spacedim>>(
        std::forward<Args>(args)...);
    */

    Assert(false, dealii::ExcNotImplemented());
    return std::unique_ptr<Problem::Base>();
  }
} // namespace Factory


#endif
