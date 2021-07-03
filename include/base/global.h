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

#ifndef base_global_h
#define base_global_h


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/timer.h>


// Use 'construct on first use' idiom for global static variables.
//
// As we need to wait for mpi initialization to finish before calling
// `this_mpi_process`, we need these singletons. TimerOutput copies the
// ostream object and does not take a reference. We could set the condition for
// the ConditionalOStream object after construction, but no longer once passed
// to TimerOutput.

dealii::ConditionalOStream &
getPCOut();

dealii::TimerOutput &
getTimer();

dealii::TableHandler &
getTable();


#endif
