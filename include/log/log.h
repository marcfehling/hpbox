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

#ifndef log_log_h
#define log_log_h


#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>


namespace Log
{
  void
  log_timings();

  template <int dim, typename T, int spacedim = dim>
  void
  log_hp_diagnostics(
    const dealii::parallel::distributed::Triangulation<dim, spacedim>
                                            &triangulation,
    const dealii::DoFHandler<dim, spacedim> &dof_handler,
    const dealii::AffineConstraints<T>      &constraints);
} // namespace Log


#endif
