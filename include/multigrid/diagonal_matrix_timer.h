// ---------------------------------------------------------------------
//
// Copyright (C) 2022 - 2023 by the deal.II authors
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

#ifndef multigrid_inverse_diagonal_matrix_h
#define multigrid_inverse_diagonal_matrix_h


#include <deal.II/lac/diagonal_matrix.h>

#include <global.h>


template <typename VectorType>
class DiagonalMatrixTimer : public dealii::Subscriptor
{
public:
  DiagonalMatrixTimer(const std::string timer_section_name = "vmult_diagonal")
    : timer_section_name(timer_section_name){};

  VectorType &
  get_vector()
  {
    return inverse_diagonal.get_vector();
  };

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    dealii::TimerOutput::Scope t(getTimer(), timer_section_name);

    inverse_diagonal.vmult(dst, src);
  }

  void
  precondition_Jacobi(VectorType                           &dst,
                      const VectorType                     &src,
                      const typename VectorType::value_type omega) const
  {
    vmult(dst, src);
    dst *= omega;
  };

private:
  const std::string timer_section_name;

  dealii::DiagonalMatrix<VectorType> inverse_diagonal;
};

#endif