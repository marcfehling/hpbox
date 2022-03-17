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

#include <deal.II/hp/refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include <adaptation/p.h>
#include <base/global.h>
#include <base/linear_algebra.h>

using namespace dealii;


namespace Adaptation
{
  template <int dim, typename VectorType, int spacedim>
  p<dim, VectorType, spacedim>::p(
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
    , cell_weights(dof_handler,
                   parallel::CellWeights<dim>::ndofs_weighting(
                     {prm.weighting_factor, prm.weighting_exponent}))
  {
    Assert(prm.min_h_level <= prm.max_h_level,
           ExcMessage(
             "Triangulation level limits have been incorrectly set up."));
    Assert(prm.min_p_degree <= prm.max_p_degree,
           ExcMessage("FECollection degrees have been incorrectly set up."));

    for (unsigned int degree = 1; degree <= prm.max_p_degree; ++degree)
      face_quadrature_collection.push_back(QGauss<dim - 1>(degree + 1));

    // limit p-level difference
    if (prm.max_p_level_difference > 0)
      {
        const unsigned int min_fe_index = prm.min_p_degree - 1;
        triangulation.signals.post_p4est_refinement.connect(
          [&, min_fe_index]() {
            const parallel::distributed::TemporarilyMatchRefineFlags<dim,
                                                                     spacedim>
              refine_modifier(triangulation);
            hp::Refinement::limit_p_level_difference(dof_handler,
                                                     prm.max_p_level_difference,
                                                     /*contains=*/min_fe_index);
          });
      }
  }



  template <int dim, typename VectorType, int spacedim>
  void
  p<dim, VectorType, spacedim>::estimate_mark()
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

    hp::Refinement::full_p_adaptivity(*dof_handler);

    // manually remove all h-flags
    for (const auto &cell : triangulation->active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->clear_refine_flag();
          cell->clear_coarsen_flag();
        }
  }



  template <int dim, typename VectorType, int spacedim>
  void
  p<dim, VectorType, spacedim>::refine()
  {
    TimerOutput::Scope t(getTimer(), "refine");
    triangulation->execute_coarsening_and_refinement();
  }



  template <int dim, typename VectorType, int spacedim>
  void
  p<dim, VectorType, spacedim>::prepare_for_serialization()
  {}



  template <int dim, typename VectorType, int spacedim>
  void
  p<dim, VectorType, spacedim>::unpack_after_serialization()
  {}



  template <int dim, typename VectorType, int spacedim>
  unsigned int
  p<dim, VectorType, spacedim>::get_n_cycles() const
  {
    return prm.n_cycles;
  }



  template <int dim, typename VectorType, int spacedim>
  unsigned int
  p<dim, VectorType, spacedim>::get_n_initial_refinements() const
  {
    return prm.min_h_level;
  }



  template <int dim, typename VectorType, int spacedim>
  const Vector<double> &
  p<dim, VectorType, spacedim>::get_error_estimates() const
  {
    return error_estimates;
  }



  template <int dim, typename VectorType, int spacedim>
  const Vector<float> &
  p<dim, VectorType, spacedim>::get_hp_indicators() const
  {
    return dummy;
  }



  // explicit instantiations
  template class p<2, LinearAlgebra::distributed::Vector<double>, 2>;
  template class p<3, LinearAlgebra::distributed::Vector<double>, 3>;

#ifdef DEAL_II_WITH_TRILINOS
  template class p<2, TrilinosWrappers::MPI::BlockVector, 2>;
  template class p<3, TrilinosWrappers::MPI::BlockVector, 3>;
  template class p<2, TrilinosWrappers::MPI::Vector, 2>;
  template class p<3, TrilinosWrappers::MPI::Vector, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
  template class p<2, PETScWrappers::MPI::BlockVector, 2>;
  template class p<3, PETScWrappers::MPI::BlockVector, 3>;
  template class p<2, PETScWrappers::MPI::Vector, 2>;
  template class p<3, PETScWrappers::MPI::Vector, 3>;
#endif

} // namespace Adaptation
