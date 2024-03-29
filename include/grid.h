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

#ifndef grid_h
#define grid_h


#include <deal.II/base/exceptions.h>

#include <deal.II/grid/tria.h>


namespace Grid
{
  template <int dim, int spacedim = dim>
  void
  reentrant_corner(dealii::Triangulation<dim, spacedim> &triangulation);

  template <int dim, int spacedim = dim>
  void
  kovasznay(dealii::Triangulation<dim, spacedim> &triangulation);

  template <int dim, int spacedim = dim>
  void
  y_pipe(dealii::Triangulation<dim, spacedim> &triangulation);
} // namespace Grid


#endif
