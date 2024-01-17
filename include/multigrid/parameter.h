// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2024 by the deal.II authors
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

#ifndef multigrid_parameter_h
#define multigrid_parameter_h


#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <string>


// ----------------------------------------
// NOTE:
// This section is modified from WIP PR #11699 which determines the interface
// of the MGSolverOperatorBase class.


DEAL_II_NAMESPACE_OPEN

struct MGSolverParameters
{
  struct
  {
    std::string  type            = "cg_with_amg";
    unsigned int maxiter         = 10000;
    double       abstol          = 1e-20;
    double       reltol          = 1e-4;
    unsigned int smoother_sweeps = 1;
    unsigned int n_cycles        = 1;
    std::string  smoother_type   = "ILU";
  } coarse_solver;

  struct
  {
    std::string  type                = "chebyshev";
    double       smoothing_range     = 20;
    unsigned int degree              = 5;
    unsigned int eig_cg_n_iterations = 20;
  } smoother;

  struct
  {
    MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType p_sequence =
      MGTransferGlobalCoarseningTools::PolynomialCoarseningSequenceType::decrease_by_one;
    bool perform_h_transfer = true;
  } transfer;
};

DEAL_II_NAMESPACE_CLOSE


#endif
