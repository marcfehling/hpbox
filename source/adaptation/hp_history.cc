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

#include <deal.II/numerics/adaptation_strategies.h>
#include <deal.II/numerics/error_estimator.h>

#include <adaptation/hp_history.h>

using namespace dealii;


namespace Adaptation
{
  template <int dim, typename VectorType, int spacedim>
  hpHistory<dim, VectorType, spacedim>::hpHistory(
    const Parameters &prm,
    const VectorType &locally_relevant_solution,
    const hp::FECollection<dim, spacedim> & /*fe_collection*/,
    DoFHandler<dim, spacedim> &                          dof_handler,
    parallel::distributed::Triangulation<dim, spacedim> &triangulation)
    : prm(prm)
    , locally_relevant_solution(&locally_relevant_solution)
    , dof_handler(&dof_handler)
    , triangulation(&triangulation)
    , cell_weights(dof_handler,
                   parallel::CellWeights<dim>::ndofs_weighting(
                     {prm.weighting_factor, prm.weighting_exponent}))
    , data_transfer(
        triangulation,
        /*transfer_variable_size_data=*/false,
        &AdaptationStrategies::Refinement::l2_norm<dim, spacedim, float>,
        &AdaptationStrategies::Coarsening::l2_norm<dim, spacedim, float>)
    , init_step(true)
  {
    Assert(prm.min_h_level <= prm.max_h_level,
           ExcMessage(
             "Triangulation level limits have been incorrectly set up."));
    Assert(prm.min_p_degree <= prm.max_p_degree,
           ExcMessage("FECollection degrees have been incorrectly set up."));

    for (unsigned int degree = prm.min_p_degree; degree <= prm.max_p_degree;
         ++degree)
      face_quadrature_collection.push_back(QGauss<dim - 1>(degree + 1));

    // limit p-level difference *before* predicting errors
    triangulation.signals.post_p4est_refinement.connect([&]() {
      const parallel::distributed::TemporarilyMatchRefineFlags<dim, spacedim>
        refine_modifier(triangulation);

      hp::Refinement::limit_p_level_difference(dof_handler,
                                               prm.max_p_level_difference,
                                               /*contains_fe_index=*/0);

      error_predictions.reinit(triangulation.n_active_cells());
      hp::Refinement::predict_error(dof_handler,
                                    error_estimates,
                                    error_predictions,
                                    /*gamma_p=*/std::sqrt(0.4),
                                    /*gamma_h=*/2.,
                                    /*gamma_n=*/1.);
    });
  }



  template <int dim, typename VectorType, int spacedim>
  void
  hpHistory<dim, VectorType, spacedim>::estimate_mark_refine()
  {
    // error estimates
    error_estimates.grow_or_shrink(triangulation->n_active_cells());

    KellyErrorEstimator<dim>::estimate(
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

    if (init_step)
      {
        // flag cells
        for (const auto &cell : triangulation->active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_refine_flag();

        init_step = false;
      }
    else
      {
        // flag cells
        parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
          *triangulation,
          error_estimates,
          prm.total_refine_fraction,
          prm.total_coarsen_fraction);

        // hp-indicators
        hp_indicators.grow_or_shrink(triangulation->n_active_cells());

        for (unsigned int i = 0; i < triangulation->n_active_cells(); ++i)
          hp_indicators(i) = error_predictions(i) - error_estimates(i);

        const float global_minimum =
          Utilities::MPI::min(*std::min_element(hp_indicators.begin(),
                                                hp_indicators.end()),
                              triangulation->get_communicator());
        if (global_minimum <= 0)
          {
            // (- 1.) ensures that the smallest indicator will be > 0
            const float decrement = global_minimum - 1.;
            for (auto &indicator : hp_indicators)
              indicator -= decrement;
          }

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

    // perform refinement
    data_transfer.prepare_for_coarsening_and_refinement(error_estimates);

    triangulation->execute_coarsening_and_refinement();

    error_predictions.grow_or_shrink(triangulation->n_active_cells());
    data_transfer.unpack(error_predictions);
  }



  template <int dim, typename VectorType, int spacedim>
  const Vector<float> &
  hpHistory<dim, VectorType, spacedim>::get_error_estimates() const
  {
    return error_estimates;
  }



  template <int dim, typename VectorType, int spacedim>
  const Vector<float> &
  hpHistory<dim, VectorType, spacedim>::get_hp_indicators() const
  {
    return hp_indicators;
  }



  // explicit instantiations
  template class hpHistory<2>;
  template class hpHistory<3>;
} // namespace Adaptation
