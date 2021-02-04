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

#ifndef adaptation_factory_h
#define adaptation_factory_h


#include <deal.II/base/exceptions.h>

#include <adaptation/base.h>
#include <adaptation/h.h>
#include <adaptation/hp_fourier.h>
#include <adaptation/hp_history.h>
#include <adaptation/hp_legendre.h>


namespace Factory
{
  template <
    int dim,
    typename VectorType = dealii::LinearAlgebra::distributed::Vector<double>,
    int spacedim        = dim,
    typename... Args>
  std::unique_ptr<Adaptation::Base>
  create_adaptation(std::string type, Args &...args)
  {
    if (type == "h")
      return std::make_unique<Adaptation::h<dim, VectorType, spacedim>>(
        args...);
    else if (type == "hp Fourier")
      return std::make_unique<Adaptation::hpFourier<dim, VectorType, spacedim>>(
        args...);
    else if (type == "hp History")
      return std::make_unique<Adaptation::hpHistory<dim, VectorType, spacedim>>(
        args...);
    else if (type == "hp Legendre")
      return std::make_unique<
        Adaptation::hpLegendre<dim, VectorType, spacedim>>(args...);
    else
      Assert(false, dealii::ExcNotImplemented());
  }
} // namespace Factory


#endif
