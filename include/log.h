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

#ifndef log_h
#define log_h


#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/solver_control.h>

#include <parameter.h>


namespace Log
{
  void
  log_cycle(const unsigned int cycle, const Parameter &prm);

  template <int dim, typename T, int spacedim = dim>
  void
  log_hp_diagnostics(
    const dealii::parallel::distributed::Triangulation<dim, spacedim> &triangulation,
    const dealii::DoFHandler<dim, spacedim>                           &dof_handler,
    const dealii::AffineConstraints<T>                                &constraint);

  template <int dim, typename T, int spacedim = dim>
  void
  log_hp_diagnostics(
    const dealii::parallel::distributed::Triangulation<dim, spacedim> &triangulation,
    const std::vector<const dealii::DoFHandler<dim, spacedim> *>      &dof_handlers,
    const std::vector<const dealii::AffineConstraints<T> *>           &constraints);

  void
  log_iterations(const dealii::SolverControl &control);

  template <typename MatrixType>
  void
  log_nonzero_elements(const MatrixType &matrix);

  void
  log_timings();
} // namespace Log


#endif
