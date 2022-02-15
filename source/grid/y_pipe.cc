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

#include <grid/y_pipe.h>

using namespace dealii;


namespace Grid
{
  template <int dim, int spacedim>
  void
  y_pipe(Triangulation<dim, spacedim> &triangulation)
  {
    const std::vector<std::pair<Point<spacedim>, double>> openings = {
      {{{-4., 0., 0.}, 1.},
       {{1, 1*std::sqrt(3.), 0.}, 1.},
       {{1, -1*std::sqrt(3.), 0.}, 1.}}};

    const std::pair<Point<spacedim>, double> bifurcation = {{0., 0., 0.}, 1.};

    GridGenerator::pipe_junction(triangulation, openings, bifurcation);
  }



  // explicit instantiations
  template void
  y_pipe<2, 2>(Triangulation<2, 2> &);
  template void
  y_pipe<3, 3>(Triangulation<3, 3> &);
} // namespace Grid
