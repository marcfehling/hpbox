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

#ifndef adaptation_base_h
#define adaptation_base_h


#include <deal.II/lac/vector.h>


namespace Adaptation
{
  class Base
  {
  public:
    virtual void
    estimate_mark_refine() = 0;
    virtual const dealii::Vector<float> &
    get_error_estimates() const = 0;
    virtual const dealii::Vector<float> &
    get_hp_indicators() const = 0;
  };
} // namespace Adaptation


#endif
