// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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

#include <function/reentrant_corner.h>

#include <cmath>

using namespace dealii;


template <int dim>
ReentrantCorner<dim>::ReentrantCorner(const double alpha)
  : Function<dim>()
  , alpha(alpha)
{
  Assert(dim > 1, ExcNotImplemented());
  Assert(alpha > 0, ExcLowerRange(alpha, 0));
}



template <int dim>
double
ReentrantCorner<dim>::value(const dealii::Point<dim> &p,
                            const unsigned int /*component*/) const
{
  const std::array<double, dim> p_sphere =
    GeometricUtilities::Coordinates::to_spherical(p);

  return std::pow(p_sphere[0], alpha) * std::sin(alpha * p_sphere[1]);
}



template <int dim>
Tensor<1, dim>
ReentrantCorner<dim>::gradient(const dealii::Point<dim> &p,
                               const unsigned int /*component*/) const
{
  const std::array<double, dim> p_sphere =
    GeometricUtilities::Coordinates::to_spherical(p);

  std::array<double, dim> ret_sphere;
  // only for polar coordinates
  const double fac = alpha * std::pow(p_sphere[0], alpha - 1);
  ret_sphere[0]    = fac * std::sin(alpha * p_sphere[1]);
  ret_sphere[1]    = fac * std::cos(alpha * p_sphere[1]);

  // transform back to cartesian coordinates
  // by considering polar unit vectors
  Tensor<1, dim> ret;
  ret[0] = ret_sphere[0] * std::cos(p_sphere[1]) -
           ret_sphere[1] * std::sin(p_sphere[1]);
  ret[1] = ret_sphere[0] * std::sin(p_sphere[1]) +
           ret_sphere[1] * std::cos(p_sphere[1]);
  return ret;
}



// explicit instantiations
template class ReentrantCorner<2>;
template class ReentrantCorner<3>;
