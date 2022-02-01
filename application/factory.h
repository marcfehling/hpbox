// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
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

#ifndef application_factory_h
#define application_factory_h


#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <base/linear_algebra.h>
#include <problem/factory.h>


namespace Factory
{
  template <typename... Args>
  std::unique_ptr<Problem::Base>
  create_application(const std::string  &type,
                     const unsigned int dimension,
                     const std::string  &linear_algebra,
                     Args &&...args)
  {
    if (linear_algebra == "dealii & Trilinos")
      {
#ifdef DEAL_II_WITH_TRILINOS
        if (dimension == 2)
          return create_problem<2, dealiiTrilinos, 2>(
            type, std::forward<Args>(args)...);
        else if (dimension == 3)
          return create_problem<3, dealiiTrilinos, 3>(
            type, std::forward<Args>(args)...);
        else
          Assert(false, dealii::ExcNotImplemented());
#else
        Assert(false,
               dealii::ExcMessage(
                 "deal.II has not been configured with Trilinos!"));
#endif
      }
    else if (linear_algebra == "Trilinos")
      {
#ifdef DEAL_II_WITH_TRILINOS
        if (dimension == 2)
          return create_problem<2, Trilinos, 2>(type,
                                                std::forward<Args>(args)...);
        else if (dimension == 3)
          return create_problem<3, Trilinos, 3>(type,
                                                std::forward<Args>(args)...);
        else
          Assert(false, dealii::ExcNotImplemented());
#else
        Assert(false,
               dealii::ExcMessage(
                 "deal.II has not been configured with Trilinos!"));
#endif
      }
    else if (linear_algebra == "PETSc")
      {
#ifdef DEAL_II_WITH_PETSC
        if (dimension == 2)
          return create_problem<2, PETSc, 2>(type, std::forward<Args>(args)...);
        else if (dimension == 3)
          return create_problem<3, PETSc, 3>(type, std::forward<Args>(args)...);
        else
          Assert(false, dealii::ExcNotImplemented());
#else
        Assert(false,
               dealii::ExcMessage(
                 "deal.II has not been configured with PETSc!"));
#endif
      }

    Assert(false, dealii::ExcNotImplemented());
    return std::unique_ptr<Problem::Base>();
  }
} // namespace Factory


#endif
