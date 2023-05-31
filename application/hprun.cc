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


#include <base/global.h>
// #include <factory.h>
#include <parameter.h>

#include <base/linear_algebra.h>
#include <poisson/matrixbased/problem.h>
#include <poisson/matrixfree/problem.h>
#include <stokes/matrixbased/problem.h>
#include <stokes/matrixfree/problem.h>

namespace Factory
{
  template <int dim, typename LinearAlgebra, int spacedim = dim, typename... Args>
  std::unique_ptr<ProblemInterface>
  create_problem(const std::string &problem_type, const std::string operator_type, Args &&...args)
  {
    if (problem_type == "Poisson")
      {
        if (operator_type == "MatrixBased")
          return std::make_unique<Poisson::MatrixBased::Problem<dim, LinearAlgebra, spacedim>>(
            std::forward<Args>(args)...);
//        else if (operator_type == "MatrixFree")
//          return std::make_unique<Poisson::Problem<dim, LinearAlgebra, spacedim>>(
//            std::forward<Args>(args)...);
      }
    else if (problem_type == "Stokes")
      {
        return std::make_unique<StokesMatrixBased::Problem<dim, LinearAlgebra, spacedim>>(
          std::forward<Args>(args)...);
      }
    else if (problem_type == "StokesMatrixFree")
      {
        if constexpr (std::is_same_v<LinearAlgebra, dealiiTrilinos>)
          {
            return std::make_unique<StokesMatrixFree::Problem<dim, LinearAlgebra, spacedim>>(
                std::forward<Args>(args)...);
          }
        else
          {
            AssertThrow(false, dealii::ExcMessage("MatrixFree only available with dealii & Trilinos!"));
          }
      }

    AssertThrow(false, dealii::ExcNotImplemented());
    return std::unique_ptr<ProblemInterface>();
  }



  template <typename... Args>
  std::unique_ptr<ProblemInterface>
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
          return create_problem<2, dealiiTrilinos, 2>(problem_type, operator_type, std::forward<Args>(args)...);
        else if (dimension == 3)
          return create_problem<3, dealiiTrilinos, 3>(problem_type, operator_type, std::forward<Args>(args)...);
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
          return create_problem<2, Trilinos, 2>(problem_type, operator_type, std::forward<Args>(args)...);
        else if (dimension == 3)
          return create_problem<3, Trilinos, 3>(problem_type, operator_type, std::forward<Args>(args)...);
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
          return create_problem<2, PETSc, 2>(problem_type, operator_type, std::forward<Args>(args)...);
        else if (dimension == 3)
          return create_problem<3, PETSc, 3>(problem_type, operator_type, std::forward<Args>(args)...);
        else
          AssertThrow(false, dealii::ExcNotImplemented());
#else
        AssertThrow(false, dealii::ExcMessage("deal.II has not been configured with PETSc!"));
#endif
      }

    AssertThrow(false, dealii::ExcNotImplemented());
    return std::unique_ptr<ProblemInterface>();
  }
}


int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                                  argv,
                                                                  1);

      Parameter prm;

      const std::string filename        = (argc > 1) ? argv[1] : "";
      const std::string output_filename = (argc > 1) ? "" : "poisson.prm";
      dealii::ParameterAcceptor::initialize(filename, output_filename);

      getPCOut() << "Running with " << prm.linear_algebra << " on "
                 << dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                 << " MPI rank(s)..." << std::endl;

      std::unique_ptr<ProblemInterface> problem = Factory::create_application(
        prm.problem_type, prm.operator_type, prm.dimension, prm.linear_algebra, prm);
      problem->run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
