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


#include <deal.II/lac/vector.h>

#include <function/kovasznay.h>

#include <cmath>

using namespace dealii;


namespace Function
{
  template <int dim>
  KovasznayExact<dim>::KovasznayExact()
    : dealii::Function<dim>(dim + 1)
  {
    Assert(dim == 2, ExcNotImplemented());
  }



  template <int dim>
  void
  KovasznayExact<dim>::vector_value(const Point<dim> &p,
                               Vector<double> &  values) const
  {
    const double &R_x = p[0];
    const double &R_y = p[1];

    // TODO: Change to std implementation
    values[0] =
      -exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * cos(2 * R_y * pi) + 1;

    values[1] = (1.0L / 2.0L) * (-sqrt(25.0 + 4 * pi2) + 5.0) *
                exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * sin(2 * R_y * pi) /
                pi;

    values[2] =
      -1.0L / 2.0L * exp(R_x * (-2 * sqrt(25.0 + 4 * pi2) + 10.0)) -
      2.0 *
        (-6538034.74494422 +
         0.0134758939981709 * exp(4 * sqrt(25.0 + 4 * pi2))) /
        (-80.0 * exp(3 * sqrt(25.0 + 4 * pi2)) +
         16.0 * sqrt(25.0 + 4 * pi2) * exp(3 * sqrt(25.0 + 4 * pi2))) -
      1634508.68623606 * exp(-3.0 * sqrt(25.0 + 4 * pi2)) /
        (-10.0 + 2.0 * sqrt(25.0 + 4 * pi2)) +
      (-0.00673794699908547 * exp(sqrt(25.0 + 4 * pi2)) +
       3269017.37247211 * exp(-3 * sqrt(25.0 + 4 * pi2))) /
        (-8 * sqrt(25.0 + 4 * pi2) + 40.0) +
      0.00336897349954273 * exp(1.0 * sqrt(25.0 + 4 * pi2)) /
        (-10.0 + 2.0 * sqrt(25.0 + 4 * pi2));
  }



  template <int dim>
  KovasznayRHS<dim>::KovasznayRHS()
    : dealii::Function<dim>(dim + 1)
  {
    Assert(dim == 2, ExcNotImplemented());
  }



  template <int dim>
  void
  KovasznayRHS<dim>::vector_value(const Point<dim> &p,
                               Vector<double> &  values) const
  {
    const double &R_x = p[0];
    const double &R_y = p[1];

    // TODO: Change to std implementation
    values[0] =
      -1.0L / 2.0L * (-2 * sqrt(25.0 + 4 * pi2) + 10.0) *
        exp(R_x * (-2 * sqrt(25.0 + 4 * pi2) + 10.0)) -
      0.4 * pi2 * exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * cos(2 * R_y * pi) +
      0.1 * pow(-sqrt(25.0 + 4 * pi2) + 5.0, 2) *
        exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * cos(2 * R_y * pi);

    values[1] = 0.2 * pi * (-sqrt(25.0 + 4 * pi2) + 5.0) *
                  exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * sin(2 * R_y * pi) -
                0.05 * pow(-sqrt(25.0 + 4 * pi2) + 5.0, 3) *
                  exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * sin(2 * R_y * pi) /
                  pi;

    values[2] = 0;
  }



  // explicit instantiations
  template class KovasznayExact<2>;
  template class KovasznayExact<3>;
  template class KovasznayRHS<2>;
  template class KovasznayRHS<3>;
} // namespace Function
