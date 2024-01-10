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

#ifndef parameter_h
#define parameter_h


#include <deal.II/base/parameter_acceptor.h>

#include <adaptation/parameter.h>


struct Parameter : public dealii::ParameterAcceptor
{
  Parameter()
    : dealii::ParameterAcceptor()
  {
    std::string *subsection = const_cast<std::string *>(&section_name);


    *subsection = "problem";

    dimension = 2;
    add_parameter("dimension", dimension);

    linear_algebra = "dealii & Trilinos";
    add_parameter("linear algebra", linear_algebra);

    problem_type = "Poisson";
    add_parameter("problem type", problem_type);

    adaptation_type = "hp Legendre";
    add_parameter("adaptation type", adaptation_type);

    grid_type = "reentrant corner";
    add_parameter("grid type", grid_type);

    operator_type = "MatrixFree";
    add_parameter("operator type", operator_type);

    smoother_preconditioner_type = "Extended Diagonal";
    add_parameter("smoother preconditioner type", smoother_preconditioner_type);

    solver_type = "GMG";
    add_parameter("solver type", solver_type);

    solver_tolerance_factor = 1e-12;
    add_parameter("solver tolerance factor", solver_tolerance_factor);

    solver_a_expensive = false;
    add_parameter("solver A expensive", solver_a_expensive);

    solver_schur_expensive = true;
    add_parameter("solver Schur expensive", solver_schur_expensive);


    *subsection = "input output";

    file_stem = "my_problem";
    add_parameter("file stem", file_stem);

    output_frequency = 1;
    add_parameter("output each n steps", output_frequency);

    resume_filename = "";
    add_parameter("resume from filename", resume_filename);

    checkpoint_frequency = 0;
    add_parameter("checkpoint each n steps", checkpoint_frequency);

    log_nonzero_elements = false;
    add_parameter("log nonzero elements", log_nonzero_elements);
  }

  unsigned int dimension;
  std::string  linear_algebra;

  std::string problem_type;
  std::string adaptation_type;
  std::string grid_type;
  std::string operator_type;
  std::string smoother_preconditioner_type;
  std::string solver_type;
  double      solver_tolerance_factor;
  bool solver_a_expensive;
  bool solver_schur_expensive;

  std::string  file_stem;
  unsigned int output_frequency;
  std::string  resume_filename;
  unsigned int checkpoint_frequency;
  bool         log_nonzero_elements;

  Adaptation::Parameter prm_adaptation;
};


#endif
