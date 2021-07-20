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


#include <base/global.h>
#include <problem/parameter.h>
#include <problem/poisson.h>

#include "factory.h"


int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                                  argv,
                                                                  1);

      Problem::Parameters prm_problem;

      const std::string filename        = (argc > 1) ? argv[1] : "";
      const std::string output_filename = (argc > 1) ? "" : "poisson.prm";
      dealii::ParameterAcceptor::initialize(filename, output_filename);

      getPCOut() << "Running with " << prm_problem.linear_algebra << " on "
                 << dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                 << " MPI rank(s)..." << std::endl;

      std::unique_ptr<Problem::Base> problem =
        Factory::create_application("poisson",
                                    prm_problem.dimension,
                                    prm_problem.linear_algebra,
                                    prm_problem);
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
