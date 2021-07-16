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
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/hp/refinement.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/smoothness_estimator.h>

#include <adaptation/hp_fourier.h>
#include <base/explicitely_instantiate.h>
#include <base/global.h>
#include <base/linear_algebra.h>

using namespace dealii;


namespace Adaptation
{
  template <int dim, typename LinearAlgebra, int spacedim>
  hpFourier<dim, LinearAlgebra, spacedim>::hpFourier(
    const Parameters &                     prm,
    const typename LinearAlgebra::Vector & locally_relevant_solution,
    const hp::FECollection<dim, spacedim> &fe_collection,
    DoFHandler<dim, spacedim> &            dof_handler,
    parallel::distributed::Triangulation<dim, spacedim> &triangulation)
    : prm(prm)
    , locally_relevant_solution(&locally_relevant_solution)
    , dof_handler(&dof_handler)
    , triangulation(&triangulation)
    , cell_weights(dof_handler,
                   parallel::CellWeights<dim>::ndofs_weighting(
                     {prm.weighting_factor, prm.weighting_exponent}))
    , fourier(SmoothnessEstimator::Fourier::default_fe_series(fe_collection))
  {
    Assert(prm.min_h_level <= prm.max_h_level,
           ExcMessage(
             "Triangulation level limits have been incorrectly set up."));
    Assert(prm.min_p_degree <= prm.max_p_degree,
           ExcMessage("FECollection degrees have been incorrectly set up."));

    for (unsigned int degree = 1; degree <= prm.max_p_degree; ++degree)
      face_quadrature_collection.push_back(QGauss<dim - 1>(degree + 1));

    // limit p-level difference
    const unsigned int min_fe_index = prm.min_p_degree - 1;
    triangulation.signals.post_p4est_refinement.connect([&, min_fe_index]() {
      const parallel::distributed::TemporarilyMatchRefineFlags<dim, spacedim>
        refine_modifier(triangulation);
      hp::Refinement::limit_p_level_difference(dof_handler,
                                               prm.max_p_level_difference,
                                               /*contains=*/min_fe_index);
    });

    {
      TimerOutput::Scope t(getTimer(), "calculate_transformation");
      fourier.precalculate_all_transformation_matrices();
    }
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  hpFourier<dim, LinearAlgebra, spacedim>::estimate_mark()
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
      /*component_mask=*/ComponentMask(),
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

    // hp-indicators
    hp_indicators.grow_or_shrink(triangulation->n_active_cells());

    SmoothnessEstimator::Fourier::coefficient_decay(
      fourier,
      *dof_handler,
      *locally_relevant_solution,
      hp_indicators,
      /*regression_strategy=*/VectorTools::Linfty_norm,
      /*smallest_abs_coefficient=*/1e-10,
      /*only_flagged_cells=*/true);

    // decide hp
    hp::Refinement::p_adaptivity_fixed_number(*dof_handler,
                                              hp_indicators,
                                              prm.p_refine_fraction,
                                              prm.p_coarsen_fraction);
    hp::Refinement::choose_p_over_h(*dof_handler);

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



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  hpFourier<dim, LinearAlgebra, spacedim>::refine()
  {
    TimerOutput::Scope t(getTimer(), "refine");
    triangulation->execute_coarsening_and_refinement();
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  hpFourier<dim, LinearAlgebra, spacedim>::prepare_for_serialization()
  {}



  template <int dim, typename LinearAlgebra, int spacedim>
  void
  hpFourier<dim, LinearAlgebra, spacedim>::unpack_after_serialization()
  {}



  template <int dim, typename LinearAlgebra, int spacedim>
  unsigned int
  hpFourier<dim, LinearAlgebra, spacedim>::get_n_cycles() const
  {
    return prm.n_cycles;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  unsigned int
  hpFourier<dim, LinearAlgebra, spacedim>::get_n_initial_refinements() const
  {
    return prm.min_h_level;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  const Vector<float> &
  hpFourier<dim, LinearAlgebra, spacedim>::get_error_estimates() const
  {
    return error_estimates;
  }



  template <int dim, typename LinearAlgebra, int spacedim>
  const Vector<float> &
  hpFourier<dim, LinearAlgebra, spacedim>::get_hp_indicators() const
  {
    return hp_indicators;
  }



  EXPLICITLY_INSTANTIATE(hpFourier)
} // namespace Adaptation
