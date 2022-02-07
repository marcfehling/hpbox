// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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


#include <deal.II/grid/grid_generator.h>

#include <grid/kovasznay.h>

using namespace dealii;


namespace Grid
{
  template <int dim, int spacedim>
  void
  kovasznay(Triangulation<dim, spacedim> &triangulation)
  {
    GridGenerator::hyper_cube(triangulation, -0.5, 1.5);
  }



  // explicit instantiations
  template void
  kovasznay<2, 2>(Triangulation<2, 2> &);
  template void
  kovasznay<3, 3>(Triangulation<3, 3> &);
} // namespace Grid
