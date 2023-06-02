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


#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/hp/refinement.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/smoothness_estimator.h>

#include <adaptation/hp_legendre.h>
#include <global.h>
#include <linear_algebra.h>

using namespace dealii;


namespace Adaptation
{
  template <int dim, typename VectorType, int spacedim>
  hpLegendre<dim, VectorType, spacedim>::hpLegendre(
    const Parameter                                     &prm,
    const VectorType                                    &locally_relevant_solution,
    const hp::FECollection<dim, spacedim>               &fe_collection,
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
           ExcMessage("Triangulation level limits have been incorrectly set up."));
    Assert(prm.min_p_degree <= prm.max_p_degree,
           ExcMessage("FECollection degrees have been incorrectly set up."));
    if (fe_collection[0].n_components() > 1)
      Assert(component_mask.n_selected_components() == 1,
             ExcMessage("For vector problems, you need to specify "
                        "only one component."));

    // like SmoothnessEstimator::default_fe_series(), but with component_mask
    {
      std::vector<unsigned int> n_coefficients_per_direction;
      for (unsigned int i = 0; i < fe_collection.size(); ++i)
        n_coefficients_per_direction.push_back(fe_collection[i].degree + 2);

      hp::QCollection<dim> q_collection;
      for (unsigned int i = 0; i < fe_collection.size(); ++i)
        {
          const QGauss<dim>  quadrature(n_coefficients_per_direction[i]);
          const QSorted<dim> quadrature_sorted(quadrature);
          q_collection.push_back(quadrature_sorted);
        }

      legendre = std::make_unique<dealii::FESeries::Legendre<dim, spacedim>>(
        n_coefficients_per_direction,
        fe_collection,
        q_collection,
        component_mask.first_selected_component());
    }

    for (unsigned int degree = 1; degree <= prm.max_p_degree; ++degree)
      face_quadrature_collection.push_back(QGauss<dim - 1>(degree + 1));

    // limit p-level difference
    if (prm.max_p_level_difference > 0)
      {
        const unsigned int min_fe_index = prm.min_p_degree - 1;
        triangulation.signals.post_p4est_refinement.connect([&, min_fe_index]() {
          const parallel::distributed::TemporarilyMatchRefineFlags<dim, spacedim> refine_modifier(
            triangulation);
          hp::Refinement::limit_p_level_difference(dof_handler,
                                                   prm.max_p_level_difference,
                                                   /*contains=*/min_fe_index);
        });
      }

    {
      TimerOutput::Scope t(getTimer(), "calculate_transformation");
      legendre->precalculate_all_transformation_matrices();
    }
  }



  template <int dim, typename VectorType, int spacedim>
  void
  hpLegendre<dim, VectorType, spacedim>::estimate_mark()
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
      *triangulation, error_estimates, prm.total_refine_fraction, prm.total_coarsen_fraction);

    // hp-indicators
    hp_indicators.grow_or_shrink(triangulation->n_active_cells());

    SmoothnessEstimator::Legendre::coefficient_decay(
      *legendre,
      *dof_handler,
      *locally_relevant_solution,
      hp_indicators,
      /*regression_strategy=*/VectorTools::Linfty_norm,
      /*smallest_abs_coefficient=*/1e-10,
      /*only_flagged_cells=*/true);

    // set future fe indices
    hp::Refinement::p_adaptivity_fixed_number(*dof_handler,
                                              hp_indicators,
                                              prm.p_refine_fraction,
                                              prm.p_coarsen_fraction);

    // limit levels
    Assert(triangulation->n_levels() >= prm.min_h_level + 1 &&
             triangulation->n_levels() <= prm.max_h_level + 1,
           ExcInternalError());

    if (triangulation->n_levels() > prm.max_h_level)
      for (const auto &cell : triangulation->active_cell_iterators_on_level(prm.max_h_level))
        cell->clear_refine_flag();

    for (const auto &cell : triangulation->active_cell_iterators_on_level(prm.min_h_level))
      cell->clear_coarsen_flag();

    // decide hp
    hp::Refinement::choose_p_over_h(*dof_handler);
  }



  template <int dim, typename VectorType, int spacedim>
  void
  hpLegendre<dim, VectorType, spacedim>::refine()
  {
    TimerOutput::Scope t(getTimer(), "refine");
    triangulation->execute_coarsening_and_refinement();
  }



  template <int dim, typename VectorType, int spacedim>
  void
  hpLegendre<dim, VectorType, spacedim>::prepare_for_serialization()
  {}



  template <int dim, typename VectorType, int spacedim>
  void
  hpLegendre<dim, VectorType, spacedim>::unpack_after_serialization()
  {}



  template <int dim, typename VectorType, int spacedim>
  unsigned int
  hpLegendre<dim, VectorType, spacedim>::get_n_cycles() const
  {
    return prm.n_cycles;
  }



  template <int dim, typename VectorType, int spacedim>
  unsigned int
  hpLegendre<dim, VectorType, spacedim>::get_n_initial_refinements() const
  {
    return prm.min_h_level;
  }



  template <int dim, typename VectorType, int spacedim>
  const Vector<float> &
  hpLegendre<dim, VectorType, spacedim>::get_error_estimates() const
  {
    return error_estimates;
  }



  template <int dim, typename VectorType, int spacedim>
  const Vector<float> &
  hpLegendre<dim, VectorType, spacedim>::get_hp_indicators() const
  {
    return hp_indicators;
  }



  // explicit instantiations
  template class hpLegendre<2, LinearAlgebra::distributed::BlockVector<double>, 2>;
  template class hpLegendre<3, LinearAlgebra::distributed::BlockVector<double>, 3>;
  template class hpLegendre<2, LinearAlgebra::distributed::Vector<double>, 2>;
  template class hpLegendre<3, LinearAlgebra::distributed::Vector<double>, 3>;

#ifdef DEAL_II_WITH_TRILINOS
  template class hpLegendre<2, TrilinosWrappers::MPI::BlockVector, 2>;
  template class hpLegendre<3, TrilinosWrappers::MPI::BlockVector, 3>;
  template class hpLegendre<2, TrilinosWrappers::MPI::Vector, 2>;
  template class hpLegendre<3, TrilinosWrappers::MPI::Vector, 3>;
#endif

#ifdef DEAL_II_WITH_PETSC
  template class hpLegendre<2, PETScWrappers::MPI::BlockVector, 2>;
  template class hpLegendre<3, PETScWrappers::MPI::BlockVector, 3>;
  template class hpLegendre<2, PETScWrappers::MPI::Vector, 2>;
  template class hpLegendre<3, PETScWrappers::MPI::Vector, 3>;
#endif

} // namespace Adaptation
