// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2023 by the deal.II authors
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
// #include <deal.II/base/flow_function.h>

#include <adaptation/h.h>
#include <adaptation/hp_fourier.h>
#include <adaptation/hp_history.h>
#include <adaptation/hp_legendre.h>
#include <adaptation/p.h>
#include <function.h>
#include <grid.h>
#include <linear_algebra.h>
#include <poisson/problem.h>
#include <stokes_matrixbased/problem.h>
#include <stokes_matrixfree/problem.h>

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
      return std::make_unique<Adaptation::hpLegendre<dim, VectorType, spacedim>>(
        std::forward<Args>(args)...);

    AssertThrow(false, dealii::ExcNotImplemented());
    return std::unique_ptr<Adaptation::Base>();
  }



  template <int dim, typename... Args>
  std::unique_ptr<dealii::Function<dim>>
  create_function(const std::string &type, Args &&...args)
  {
    if (type == "zero")
      return std::make_unique<dealii::Functions::ZeroFunction<dim>>(std::forward<Args>(args)...);
    else if (type == "reentrant corner")
      return std::make_unique<Function::ReentrantCorner<dim>>(std::forward<Args>(args)...);
    else if (type == "kovasznay exact")
      return std::make_unique<Function::KovasznayExact<dim>>(std::forward<Args>(args)...);
    else if (type == "kovasznay exact velocity")
      return std::make_unique<Function::KovasznayExactVelocity<dim>>(std::forward<Args>(args)...);
    else if (type == "kovasznay exact pressure")
      return std::make_unique<Function::KovasznayExactPressure<dim>>(std::forward<Args>(args)...);
    else if (type == "kovasznay rhs")
      return std::make_unique<Function::KovasznayRHS<dim>>(std::forward<Args>(args)...);
    else if (type == "kovasznay rhs velocity")
      return std::make_unique<Function::KovasznayRHSVelocity<dim>>(std::forward<Args>(args)...);
    // else if (type == "poisseuille")
    //   return std::make_unique<dealii::Functions::PoisseuilleFlow<dim>>(std::forward<Args>(args)...);
    // else if (type == "poisseuille velocity")
    //   return std::make_unique<Function::PoisseuilleFlowVelocity<dim>>(std::forward<Args>(args)...);
    // else if (type == "poisseuille pressure")
    //   return std::make_unique<Function::PoisseuilleFlowPressure<dim>>(std::forward<Args>(args)...);
    // TODO: parameter pack needs to be same size as constructor parameters
    //       group with if constexpr sizeof(Args) == 1

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



  template <int dim, typename LinearAlgebra, int spacedim = dim, typename... Args>
  std::unique_ptr<ProblemBase>
  create_problem(const std::string &problem_type, const std::string operator_type, Args &&...args)
  {
    if (problem_type == "Poisson")
      {
        return std::make_unique<Poisson::Problem<dim, LinearAlgebra, spacedim>>(
          std::forward<Args>(args)...);
      }
    else if (problem_type == "Stokes")
      {
        if (operator_type == "MatrixBased")
          {
            return std::make_unique<StokesMatrixBased::Problem<dim, LinearAlgebra, spacedim>>(
              std::forward<Args>(args)...);
          }
        else if (operator_type == "MatrixFree")
          {
            if constexpr (std::is_same_v<LinearAlgebra, dealiiTrilinos>)
              {
                return std::make_unique<StokesMatrixFree::Problem<dim, LinearAlgebra, spacedim>>(
                  std::forward<Args>(args)...);
              }
            else
              {
                AssertThrow(
                  false, dealii::ExcMessage("MatrixFree only available with dealii & Trilinos!"));
              }
          }
      }

    AssertThrow(false, dealii::ExcNotImplemented());
    return std::unique_ptr<ProblemBase>();
  }



  template <typename... Args>
  std::unique_ptr<ProblemBase>
  create_application(const std::string &problem_type,
                     const std::string &operator_type,
                     const unsigned int dimension,
                     const std::string &linear_algebra,
                     Args &&...args)
  {
    if (linear_algebra == "dealii & Trilinos")
      {
#ifdef DEAL_II_WITH_TRILINOS
        if (dimension == 2)
          return create_problem<2, dealiiTrilinos, 2>(problem_type,
                                                      operator_type,
                                                      std::forward<Args>(args)...);
        else if (dimension == 3)
          return create_problem<3, dealiiTrilinos, 3>(problem_type,
                                                      operator_type,
                                                      std::forward<Args>(args)...);
        else
          AssertThrow(false, dealii::ExcNotImplemented());
#else
        AssertThrow(false, dealii::ExcMessage("deal.II has not been configured with Trilinos!"));
#endif
      }
    else if (linear_algebra == "Trilinos")
      {
#ifdef DEAL_II_WITH_TRILINOS
        if (dimension == 2)
          return create_problem<2, Trilinos, 2>(problem_type,
                                                operator_type,
                                                std::forward<Args>(args)...);
        else if (dimension == 3)
          return create_problem<3, Trilinos, 3>(problem_type,
                                                operator_type,
                                                std::forward<Args>(args)...);
        else
          AssertThrow(false, dealii::ExcNotImplemented());
#else
        AssertThrow(false, dealii::ExcMessage("deal.II has not been configured with Trilinos!"));
#endif
      }
    else if (linear_algebra == "PETSc")
      {
#ifdef DEAL_II_WITH_PETSC
        if (dimension == 2)
          return create_problem<2, PETSc, 2>(problem_type,
                                             operator_type,
                                             std::forward<Args>(args)...);
        else if (dimension == 3)
          return create_problem<3, PETSc, 3>(problem_type,
                                             operator_type,
                                             std::forward<Args>(args)...);
        else
          AssertThrow(false, dealii::ExcNotImplemented());
#else
        AssertThrow(false, dealii::ExcMessage("deal.II has not been configured with PETSc!"));
#endif
      }

    AssertThrow(false, dealii::ExcNotImplemented());
    return std::unique_ptr<ProblemBase>();
  }
} // namespace Factory


#endif
