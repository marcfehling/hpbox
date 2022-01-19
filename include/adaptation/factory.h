// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2022 by the deal.II authors
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

#include <adaptation/h.h>
#include <adaptation/hp_fourier.h>
#include <adaptation/hp_history.h>
#include <adaptation/hp_legendre.h>
#include <adaptation/p.h>


namespace Factory
{
  template <int dim,
            typename VectorType,
            int spacedim = dim,
            typename... Args>
  std::unique_ptr<Adaptation::Base>
  create_adaptation(const std::string &type, Args &&...args)
  {
    if (type == "h")
      return std::make_unique<Adaptation::h<dim, VectorType, spacedim>>(
        std::forward<Args>(args)...);
    else if (type == "p")
      return std::make_unique<Adaptation::p<dim, VectorType, spacedim>>(
        std::forward<Args>(args)...);
    else if (type == "hp Fourier")
      return std::make_unique<
        Adaptation::hpFourier<dim, VectorType, spacedim>>(
        std::forward<Args>(args)...);
    else if (type == "hp History")
      return std::make_unique<
        Adaptation::hpHistory<dim, VectorType, spacedim>>(
        std::forward<Args>(args)...);
    else if (type == "hp Legendre")
      return std::make_unique<
        Adaptation::hpLegendre<dim, VectorType, spacedim>>(
        std::forward<Args>(args)...);

    Assert(false, dealii::ExcNotImplemented());
    return std::unique_ptr<Adaptation::Base>();
  }
} // namespace Factory


#endif
