// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2023 by the deal.II authors
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


#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometric_utilities.h>

#include <deal.II/lac/vector.h>

#include <function.h>

#include <cmath>

using namespace dealii;


namespace Function
{
  template <int dim>
  ReentrantCorner<dim>::ReentrantCorner(const double alpha)
    : dealii::Function<dim>()
    , alpha(alpha)
  {
    Assert(dim > 1, ExcNotImplemented());
    Assert(alpha > 0, ExcLowerRange(alpha, 0));
  }

  template <int dim>
  double
  ReentrantCorner<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
  {
    const std::array<double, dim> p_sphere = GeometricUtilities::Coordinates::to_spherical(p);

    return std::pow(p_sphere[0], alpha) * std::sin(alpha * p_sphere[1]);
  }

  template <int dim>
  Tensor<1, dim>
  ReentrantCorner<dim>::gradient(const Point<dim> &p, const unsigned int /*component*/) const
  {
    const std::array<double, dim> p_sphere = GeometricUtilities::Coordinates::to_spherical(p);

    std::array<double, dim> ret_sphere;
    // only for polar coordinates
    const double fac = alpha * std::pow(p_sphere[0], alpha - 1);
    ret_sphere[0]    = fac * std::sin(alpha * p_sphere[1]);
    ret_sphere[1]    = fac * std::cos(alpha * p_sphere[1]);

    // transform back to cartesian coordinates
    // by considering polar unit vectors
    Tensor<1, dim> ret;
    ret[0] = ret_sphere[0] * std::cos(p_sphere[1]) - ret_sphere[1] * std::sin(p_sphere[1]);
    ret[1] = ret_sphere[0] * std::sin(p_sphere[1]) + ret_sphere[1] * std::cos(p_sphere[1]);
    return ret;
  }



  template <int dim>
  KovasznayExactVelocity<dim>::KovasznayExactVelocity()
    : dealii::Function<dim>(dim)
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  template <int dim>
  void
  KovasznayExactVelocity<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
  {
    const double R_x = p[0];
    const double R_y = p[1];

    constexpr double pi  = numbers::PI;
    constexpr double pi2 = numbers::PI * numbers::PI;

    values[0] = -std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) * std::cos(2 * R_y * pi) + 1;
    values[1] = (1.0L / 2.0L) * (-std::sqrt(25.0 + 4 * pi2) + 5.0) *
                std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) * std::sin(2 * R_y * pi) / pi;
  }



  template <int dim>
  KovasznayExactPressure<dim>::KovasznayExactPressure()
    : dealii::Function<dim>(1)
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  template <int dim>
  double
  KovasznayExactPressure<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
  {
    const double R_x = p[0];

    constexpr double pi2 = numbers::PI * numbers::PI;

    return -1.0L / 2.0L * std::exp(R_x * (-2 * std::sqrt(25.0 + 4 * pi2) + 10.0)) -
           2.0 *
             (-6538034.74494422 + 0.0134758939981709 * std::exp(4 * std::sqrt(25.0 + 4 * pi2))) /
             (-80.0 * std::exp(3 * std::sqrt(25.0 + 4 * pi2)) +
              16.0 * std::sqrt(25.0 + 4 * pi2) * std::exp(3 * std::sqrt(25.0 + 4 * pi2))) -
           1634508.68623606 * std::exp(-3.0 * std::sqrt(25.0 + 4 * pi2)) /
             (-10.0 + 2.0 * std::sqrt(25.0 + 4 * pi2)) +
           (-0.00673794699908547 * std::exp(std::sqrt(25.0 + 4 * pi2)) +
            3269017.37247211 * std::exp(-3 * std::sqrt(25.0 + 4 * pi2))) /
             (-8 * std::sqrt(25.0 + 4 * pi2) + 40.0) +
           0.00336897349954273 * std::exp(1.0 * std::sqrt(25.0 + 4 * pi2)) /
             (-10.0 + 2.0 * std::sqrt(25.0 + 4 * pi2));
  }



  template <int dim>
  KovasznayExact<dim>::KovasznayExact()
    : dealii::Function<dim>(dim + 1)
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  template <int dim>
  void
  KovasznayExact<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
  {
    velocity_function.vector_value(p, values);
    values[2] = pressure_function.value(p);
  }



  template <int dim>
  KovasznayRHSVelocity<dim>::KovasznayRHSVelocity()
    : dealii::Function<dim>(dim)
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  template <int dim>
  void
  KovasznayRHSVelocity<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
  {
    const double R_x = p[0];
    const double R_y = p[1];

    constexpr double pi  = numbers::PI;
    constexpr double pi2 = numbers::PI * numbers::PI;

    values[0] =
      -1.0L / 2.0L * (-2 * std::sqrt(25.0 + 4 * pi2) + 10.0) *
        std::exp(R_x * (-2 * std::sqrt(25.0 + 4 * pi2) + 10.0)) -
      0.4 * pi2 * std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) * std::cos(2 * R_y * pi) +
      0.1 * std::pow(-std::sqrt(25.0 + 4 * pi2) + 5.0, 2) *
        std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) * std::cos(2 * R_y * pi);
    values[1] = 0.2 * pi * (-std::sqrt(25.0 + 4 * pi2) + 5.0) *
                  std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) * std::sin(2 * R_y * pi) -
                0.05 * std::pow(-std::sqrt(25.0 + 4 * pi2) + 5.0, 3) *
                  std::exp(R_x * (-std::sqrt(25.0 + 4 * pi2) + 5.0)) * std::sin(2 * R_y * pi) / pi;
  }



  template <int dim>
  KovasznayRHS<dim>::KovasznayRHS()
    : dealii::Function<dim>(dim + 1)
  {
    Assert(dim == 2, ExcNotImplemented());
  }

  template <int dim>
  void
  KovasznayRHS<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
  {
    velocity_function.vector_value(p, values);
    values[2] = 0.;
  }



  template <int dim>
  PoisseuilleFlowVelocity<dim>::PoisseuilleFlowVelocity(const double r)
    : dealii::Function<dim>(dim)
    , inv_sqr_radius(1 / r / r)
  {}

  template <int dim>
  void
  PoisseuilleFlowVelocity<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
  {
    // First, compute the square of the distance to the x-axis divided by the radius.
    double r2 = 0;
    for (unsigned int d = 1; d < dim; ++d)
      r2 += p(d) * p(d);
    r2 *= inv_sqr_radius;

    // x-velocity
    values[0] = 1. - r2;
    // other velocities
    for (unsigned int d = 1; d < dim; ++d)
      values[d] = 0.;
  }



  template <int dim>
  PoisseuilleFlowPressure<dim>::PoisseuilleFlowPressure(const double r, const double Re)
    : dealii::Function<dim>(1)
    , inv_sqr_radius(1 / r / r)
    , Reynolds(Re)
  {
    Assert(Reynolds != 0., ExcMessage("Reynolds number cannot be zero"));
  }

  template <int dim>
  double
  PoisseuilleFlowPressure<dim>::value(const Point<dim> &p, const unsigned int /*component*/) const
  {
    return -2 * (dim - 1) * inv_sqr_radius * p(0) / Reynolds; // + this->mean_pressure;
  }



  // explicit instantiations
  template class ReentrantCorner<2>;
  template class ReentrantCorner<3>;
  template class KovasznayExactVelocity<2>;
  template class KovasznayExactVelocity<3>;
  template class KovasznayExactPressure<2>;
  template class KovasznayExactPressure<3>;
  template class KovasznayExact<2>;
  template class KovasznayExact<3>;
  template class KovasznayRHSVelocity<2>;
  template class KovasznayRHSVelocity<3>;
  template class KovasznayRHS<2>;
  template class KovasznayRHS<3>;
  template class PoisseuilleFlowVelocity<2>;
  template class PoisseuilleFlowVelocity<3>;
  template class PoisseuilleFlowPressure<2>;
  template class PoisseuilleFlowPressure<3>;
} // namespace Function
