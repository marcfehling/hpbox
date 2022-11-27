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

#ifndef factory_h
#define factory_h


#include <deal.II/base/exceptions.h>

#include <adaptation/h.h>
#include <adaptation/hp_fourier.h>
#include <adaptation/hp_history.h>
#include <adaptation/hp_legendre.h>
#include <adaptation/p.h>
#include <base/linear_algebra.h>
#include <function.h>
#include <grid.h>
#include <operator.h>
#include <poisson/matrixbased.h>
#include <poisson/matrixfree.h>
#include <poisson/problem.h>
#include <stokes/problem.h>

#include <memory>


namespace Factory
{
  template <int dim, typename VectorType, int spacedim = dim, typename... Args>
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
      return std::make_unique<Adaptation::hpFourier<dim, VectorType, spacedim>>(
        std::forward<Args>(args)...);
    else if (type == "hp History")
      return std::make_unique<Adaptation::hpHistory<dim, VectorType, spacedim>>(
        std::forward<Args>(args)...);
    else if (type == "hp Legendre")
      return std::make_unique<
        Adaptation::hpLegendre<dim, VectorType, spacedim>>(
        std::forward<Args>(args)...);

    AssertThrow(false, dealii::ExcNotImplemented());
    return std::unique_ptr<Adaptation::Base>();
  }



  template <int dim, typename... Args>
  std::unique_ptr<dealii::Function<dim>>
  create_function(const std::string &type, Args &&...args)
  {
    if (type == "zero")
      return std::make_unique<dealii::Functions::ZeroFunction<dim>>(
        std::forward<Args>(args)...);
    else if (type == "reentrant corner")
      return std::make_unique<Function::ReentrantCorner<dim>>(
        std::forward<Args>(args)...);
    else if (type == "kovasznay exact")
      return std::make_unique<Function::KovasznayExact<dim>>(
        std::forward<Args>(args)...);
    else if (type == "kovasznay rhs")
      return std::make_unique<Function::KovasznayRHS<dim>>(
        std::forward<Args>(args)...);

    AssertThrow(false, dealii::ExcNotImplemented());
    return std::unique_ptr<dealii::Function<dim>>();
  }



  template <typename... Args>
  void
  create_grid(std::string type, Args &&...args)
  {
    if (type == "reentrant corner")
      Grid::reentrant_corner(std::forward<Args>(args)...);
    else if (type == "kovasznay")
      Grid::kovasznay(std::forward<Args>(args)...);
    else if (type == "y-pipe")
      Grid::y_pipe(std::forward<Args>(args)...);
    else
      AssertThrow(false, dealii::ExcNotImplemented());
  }



  template <int dim,
            typename LinearAlgebra,
            int spacedim = dim,
            typename... Args>
  std::unique_ptr<OperatorBase<dim, LinearAlgebra, spacedim>>
  create_operator(const std::string type,
                  const std::string problem_type,
                  Args &&...args)
  {
    if (type == "MatrixBased")
      {
        if (problem_type == "Poisson")
          return std::make_unique<
            Poisson::OperatorMatrixBased<dim, LinearAlgebra, spacedim>>(
            std::forward<Args>(args)...);
        else if (problem_type == "Stokes")
          AssertThrow(false, dealii::ExcNotImplemented());
      }
    else if (type == "MatrixFree")
      {
        if constexpr (std::is_same<LinearAlgebra, dealiiTrilinos>::value)
          {
            if (problem_type == "Poisson")
              return std::make_unique<
                Poisson::OperatorMatrixFree<dim, LinearAlgebra, spacedim>>(
                std::forward<Args>(args)...);
            else if (problem_type == "Stokes")
              AssertThrow(false, dealii::ExcNotImplemented());
          }
        else
          {
            AssertThrow(false,
                        dealii::ExcMessage(
                          "MatrixFree only available with dealii & Trilinos!"));
          }
      }

    AssertThrow(false, dealii::ExcNotImplemented());
    return std::unique_ptr<OperatorBase<dim, LinearAlgebra, spacedim>>();
  }



  template <int dim,
            typename LinearAlgebra,
            int spacedim = dim,
            typename... Args>
  std::unique_ptr<ProblemBase>
  create_problem(const std::string &type, Args &&...args)
  {
    if (type == "Poisson")
      return std::make_unique<Poisson::Problem<dim, LinearAlgebra, spacedim>>(
        std::forward<Args>(args)...);
    else if (type == "Stokes")
      return std::make_unique<Stokes::Problem<dim, LinearAlgebra, spacedim>>(
        std::forward<Args>(args)...);

    AssertThrow(false, dealii::ExcNotImplemented());
    return std::unique_ptr<ProblemBase>();
  }



  template <typename... Args>
  std::unique_ptr<ProblemBase>
  create_application(const std::string &type,
                     const unsigned int dimension,
                     const std::string &linear_algebra,
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
          AssertThrow(false, dealii::ExcNotImplemented());
#else
        AssertThrow(false,
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
          AssertThrow(false, dealii::ExcNotImplemented());
#else
        AssertThrow(false,
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
          AssertThrow(false, dealii::ExcNotImplemented());
#else
        AssertThrow(false,
                    dealii::ExcMessage(
                      "deal.II has not been configured with PETSc!"));
#endif
      }

    AssertThrow(false, dealii::ExcNotImplemented());
    return std::unique_ptr<ProblemBase>();
  }
} // namespace Factory


#endif
