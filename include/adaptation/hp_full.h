// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the deal.II authors
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

#ifndef adaptation_hp_full_h
#define adaptation_hp_full_h


#include <deal.II/base/smartpointer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <adaptation/base.h>
#include <adaptation/parameter.h>


namespace Adaptation
{
  template <int dim, typename VectorType, int spacedim = dim>
  class hpFull : public Base
  {
  public:
    hpFull(const Parameter                                             &prm,
           const VectorType                                            &locally_relevant_solution,
           const dealii::hp::FECollection<dim, spacedim>               &fe_collection,
           dealii::DoFHandler<dim, spacedim>                           &dof_handler,
           dealii::parallel::distributed::Triangulation<dim, spacedim> &triangulation,
           const dealii::ComponentMask &component_mask = dealii::ComponentMask());

    virtual void
    estimate_mark() override;
    virtual void
    refine() override;

    virtual void
    prepare_for_serialization() override;
    virtual void
    unpack_after_serialization() override;

    virtual unsigned int
    get_n_cycles() const override;
    virtual unsigned int
    get_n_initial_refinements() const override;

    virtual const dealii::Vector<float> &
    get_error_estimates() const override;
    virtual const dealii::Vector<float> &
    get_hp_indicators() const override;

  protected:
    const Parameter &prm;

    const dealii::SmartPointer<const VectorType>                  locally_relevant_solution;
    const dealii::SmartPointer<dealii::DoFHandler<dim, spacedim>> dof_handler;
    const dealii::SmartPointer<dealii::parallel::distributed::Triangulation<dim, spacedim>>
      triangulation;

    const dealii::ComponentMask component_mask;

    dealii::hp::QCollection<dim - 1> face_quadrature_collection;

    dealii::Vector<float> error_estimates;
    dealii::Vector<float> dummy;
  };
} // namespace Adaptation


#endif
