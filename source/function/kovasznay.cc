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
                                    Vector<double>   &values) const
  {
    const double R_x = p[0];
    const double R_y = p[1];

    constexpr double pi  = numbers::PI;
    constexpr double pi2 = numbers::PI * numbers::PI;

    values[0] = -std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                  std::cos(2 * R_y * pi) +
                1;
    values[1] = (1.0L / 2.0L) * (-std::sqrt(25.0 + 4 * pi2) + 5.0) *
                std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                std::sin(2 * R_y * pi) / pi;
    values[2] =
      -1.0L / 2.0L * std::exp(R_x * (-2 * std::sqrt(25.0 + 4 * pi2) + 10.0)) -
      2.0 *
        (-6538034.74494422 +
         0.0134758939981709 * std::exp(4 * std::sqrt(25.0 + 4 * pi2))) /
        (-80.0 * std::exp(3 * std::sqrt(25.0 + 4 * pi2)) +
         16.0 * std::sqrt(25.0 + 4 * pi2) *
           std::exp(3 * std::sqrt(25.0 + 4 * pi2))) -
      1634508.68623606 * std::exp(-3.0 * std::sqrt(25.0 + 4 * pi2)) /
        (-10.0 + 2.0 * std::sqrt(25.0 + 4 * pi2)) +
      (-0.00673794699908547 * std::exp(std::sqrt(25.0 + 4 * pi2)) +
       3269017.37247211 * std::exp(-3 * std::sqrt(25.0 + 4 * pi2))) /
        (-8 * std::sqrt(25.0 + 4 * pi2) + 40.0) +
      0.00336897349954273 * std::exp(1.0 * std::sqrt(25.0 + 4 * pi2)) /
        (-10.0 + 2.0 * std::sqrt(25.0 + 4 * pi2));
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
                                  Vector<double>   &values) const
  {
    const double R_x = p[0];
    const double R_y = p[1];

    constexpr double pi  = numbers::PI;
    constexpr double pi2 = numbers::PI * numbers::PI;

    values[0] = -1.0L / 2.0L * (-2 * std::sqrt(25.0 + 4 * pi2) + 10.0) *
                  std::exp(R_x * (-2 * std::sqrt(25.0 + 4 * pi2) + 10.0)) -
                0.4 * pi2 * std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                  std::cos(2 * R_y * pi) +
                0.1 * std::pow(-std::sqrt(25.0 + 4 * pi2) + 5.0, 2) *
                  std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                  std::cos(2 * R_y * pi);
    values[1] = 0.2 * pi * (-std::sqrt(25.0 + 4 * pi2) + 5.0) *
                  std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                  std::sin(2 * R_y * pi) -
                0.05 * std::pow(-std::sqrt(25.0 + 4 * pi2) + 5.0, 3) *
                  std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) *
                  std::sin(2 * R_y * pi) / pi;
    values[2] = 0;
  }



  // explicit instantiations
  template class KovasznayExact<2>;
  template class KovasznayExact<3>;
  template class KovasznayRHS<2>;
  template class KovasznayRHS<3>;
} // namespace Function
