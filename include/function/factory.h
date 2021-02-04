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

#ifndef function_factory_h
#define function_factory_h


#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>

#include <function/reentrant_corner.h>


namespace Factory
{
  template <int dim, typename... Args>
  std::unique_ptr<dealii::Function<dim>>
  create_function(std::string type, Args &...args)
  {
    if (type == "zero")
      return std::make_unique<dealii::Functions::ZeroFunction<dim>>(args...);
    else if (type == "reentrant corner")
      return std::make_unique<ReentrantCorner<dim>>(args...);
    else
      Assert(false, dealii::ExcNotImplemented());
  }
} // namespace Factory


#endif
