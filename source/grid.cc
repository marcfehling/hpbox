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


#include <deal.II/grid/grid_generator.h>

#include <grid.h>

using namespace dealii;


namespace Grid
{
  template <int dim, int spacedim>
  void
  reentrant_corner(Triangulation<dim, spacedim> &triangulation)
  {
    std::vector<unsigned int> repetitions(dim);
    Point<dim>                bottom_left, top_right;
    for (unsigned int d = 0; d < dim; ++d)
      if (d < 2)
        {
          repetitions[d] = 2;
          bottom_left[d] = -1.;
          top_right[d]   = 1.;
        }
      else
        {
          repetitions[d] = 1;
          bottom_left[d] = 0.;
          top_right[d]   = 1.;
        }

    std::vector<int> cells_to_remove(dim, 1);
    cells_to_remove[0] = -1;

    GridGenerator::subdivided_hyper_L(
      triangulation, repetitions, bottom_left, top_right, cells_to_remove);
  }



  template <int dim, int spacedim>
  void
  kovasznay(Triangulation<dim, spacedim> &triangulation)
  {
    GridGenerator::subdivided_hyper_cube(triangulation, 2, -0.5, 1.5);
  }



  template <int dim, int spacedim>
  void
  y_pipe(Triangulation<dim, spacedim> &triangulation)
  {
    const std::vector<std::pair<Point<spacedim>, double>> openings = {
      {{{-2., 0., 0.}, 1.},
       {{1., 1. * std::sqrt(3.), 0.}, 1.},
       {{1., -1. * std::sqrt(3.), 0.}, 1.}}};

    const std::pair<Point<spacedim>, double> bifurcation = {{0., 0., 0.}, 1.};

    GridGenerator::pipe_junction(triangulation, openings, bifurcation);
  }



  // explicit instantiations
  template void
  reentrant_corner<2, 2>(Triangulation<2, 2> &);
  template void
  reentrant_corner<3, 3>(Triangulation<3, 3> &);
  template void
  kovasznay<2, 2>(Triangulation<2, 2> &);
  template void
  kovasznay<3, 3>(Triangulation<3, 3> &);
  template void
  y_pipe<2, 2>(Triangulation<2, 2> &);
  template void
  y_pipe<3, 3>(Triangulation<3, 3> &);
} // namespace Grid
