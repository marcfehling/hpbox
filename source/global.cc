// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2023 by the deal.II authors
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


#include <deal.II/base/mpi.h>

#include <global.h>

#include <iostream>

using namespace dealii;


ConditionalOStream &
getPCOut()
{
  static ConditionalOStream pcout(std::cout,
                                  (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
  return pcout;
}



TimerOutput &
getTimer()
{
  static TimerOutput computing_timer(MPI_COMM_WORLD,
                                     getPCOut(),
                                     TimerOutput::never,
                                     TimerOutput::wall_times);
  return computing_timer;
}



TableHandler &
getTable()
{
  static TableHandler table_handler;
  return table_handler;
}
