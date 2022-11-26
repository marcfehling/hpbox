// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2022 by the deal.II authors
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
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include <adaptation/h.h>
#include <base/global.h>
#include <base/linear_algebra.h>

using namespace dealii;


namespace Adaptation
{
  template <int dim, typename VectorType, int spacedim>
  h<dim, VectorType, spacedim>::h(
    const Parameter  &prm,
    const VectorType &locally_relevant_solution,
    const hp::FECollection<dim, spacedim> & /*fe_collection*/,
    DoFHandler<dim, spacedim>                           &dof_handler,
    parallel::distributed::Triangulation<dim, spacedim> &triangulation,
    const ComponentMask                                 &component_mask)
    : prm(prm)
    , locally_relevant_solution(&locally_relevant_solution)
    , dof_handler(&dof_handler)
    , triangulation(&triangulation)
    , component_mask(component_mask)
  {
    Assert(prm.min_h_level <= prm.max_h_level,
           ExcMessage(
             "Triangulation level limits have been incorrectly set up."));
    Assert(prm.min_p_degree <= prm.max_p_degree,
           ExcMessage("FECollection degrees have been incorrectly set up."));

    for (unsigned int degree = 1; degree <= prm.max_p_degree; ++degree)
      face_quadrature_collection.push_back(QGauss<dim - 1>(degree + 1));
  }



  template <int dim, typename VectorType, int spacedim>
  void
  h<dim, VectorType, spacedim>::estimate_mark()
  {
    TimerOutput::Scope t(getTimer(), "estimate_mark");

    // error estimates
    error_estimates.grow_or_shrink(triangulation->n_active_cells());

    KellyErrorEstimator<dim, spacedim>::estimate(
      *dof_handler,
      face_quadrature_collection,
      std::map<types::boundary_id, const Function<dim> *>(),
      *locally_relevant_solution,
      error_estimates,
      component_mask,
      /*coefficients=*/nullptr,
      /*n_threads=*/numbers::invalid_unsigned_int,
      /*subdomain_id=*/numbers::invalid_subdomain_id,
      /*material_id=*/numbers::invalid_material_id,
      /*strategy=*/
      KellyErrorEstimator<dim>::Strategy::face_diameter_over_twice_max_degree);

    // flag cells
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      *triangulation,
      error_estimates,
      prm.total_refine_fraction,
      prm.total_coarsen_fraction);

    // limit levels
    Assert(triangulation->n_levels() >= prm.min_h_level + 1 &&
             triangulation->n_levels() <= prm.max_h_level + 1,
           ExcInternalError());

    if (triangulation->n_levels() > prm.max_h_level)
      for (const auto &cell :
           triangulation->active_cell_iterators_on_level(prm.max_h_level))
        cell->clear_refine_flag();

    for (const auto &cell :
         triangulation->active_cell_iterators_on_level(prm.min_h_level))
      cell->clear_coarsen_flag();
  }



  template <int dim, typename VectorType, int spacedim>
  void
  h<dim, VectorType, spacedim>::prepare_for_serialization()
  {}



  template <int dim, typename VectorType, int spacedim>
  void
  h<dim, VectorType, spacedim>::unpack_after_serialization()
  {}



  template <int dim, typename VectorType, int spacedim>
  void
  h<dim, VectorType, spacedim>::refine()
  {
    TimerOutput::Scope t(getTimer(), "refine");
    triangulation->execute_coarsening_and_refinement();
  }



  template <int dim, typename VectorType, int spacedim>
  unsigned int
  h<dim, VectorType, spacedim>::get_n_cycles() const
  {
    return prm.n_cycles;
  }



  template <int dim, typename VectorType, int spacedim>
  unsigned int
  h<dim, VectorType, spacedim>::get_n_initial_refinements() const
  {
    return prm.min_h_level;
  }



  template <int dim, typename VectorType, int spacedim>
  const Vector<float> &
  h<dim, VectorType, spacedim>::get_error_estimates() const
  {
    return error_estimates;
  }



  template <int dim, typename VectorType, int spacedim>
  const Vector<float> &
  h<dim, VectorType, spacedim>::get_hp_indicators() const
  {
    return dummy;
  }



  // explicit instantiations
  // clang-format off
  template class h<2, LinearAlgebra::distributed::BlockVector<double>, 2>;
  template class h<3, LinearAlgebra::distributed::BlockVector<double>, 3>;
  template class h<2, LinearAlgebra::distributed::Vector<double>, 2>;
  template class h<3, LinearAlgebra::distributed::Vector<double>, 3>;

#ifdef DEAL_II_WITH_TRILINOS
  template class h<2, TrilinosWrappers::MPI::BlockVector, 2>;
  template class h<3, TrilinosWrappers::MPI::BlockVector, 3>;
  template class h<2, TrilinosWrappers::MPI::Vector, 2>;
  template class h<3, TrilinosWrappers::MPI::Vector, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
  template class h<2, PETScWrappers::MPI::BlockVector, 2>;
  template class h<3, PETScWrappers::MPI::BlockVector, 3>;
  template class h<2, PETScWrappers::MPI::Vector, 2>;
  template class h<3, PETScWrappers::MPI::Vector, 3>;
#endif
  // clang-format on

} // namespace Adaptation
