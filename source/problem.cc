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
//
//
//#include <deal.II/base/timer.h>
//
//#include <deal.II/lac/vector.h>
//
//#include <deal.II/numerics/data_out.h>
//#include <deal.II/numerics/vector_tools.h>
//
//#include <base/global.h>
//#include <base/log.h>
//#include <factory.h>
//#include <problem.h>
//
//#include <sstream>
//
//using namespace dealii;
//
//
//template <int dim, typename LinearAlgebra, int spacedim>
//ProblemBase<dim, LinearAlgebra, spacedim>::ProblemBase(const Parameter &prm)
//  : mpi_communicator(MPI_COMM_WORLD)
//  , prm(prm)
//  , triangulation(mpi_communicator)
//  , dof_handler(triangulation)
//{
//  TimerOutput::Scope t(getTimer(), "initialize_problem");
//
//  // prepare name for logfile
//  {
//    time_t             now = time(nullptr);
//    tm                *ltm = localtime(&now);
//    std::ostringstream oss;
//    oss << prm.file_stem << "-" << std::put_time(ltm, "%Y%m%d-%H%M%S") << ".log";
//    filename_log = oss.str();
//  }
//
//  // prepare collections
//  // mapping_collection.push_back(MappingQ1<dim, spacedim>());
//
//  // for (unsigned int degree = 1; degree <= prm.prm_adaptation.max_p_degree; ++degree)
//  //   {
//  //     fe_collection.push_back(FE_Q<dim, spacedim>(degree));
//  //     quadrature_collection.push_back(QGauss<dim>(degree + 1));
//  //   }
//
//  // prepare hierarchy
//  const unsigned int min_fe_index = prm.prm_adaptation.min_p_degree - 1;
//  fe_collection.set_hierarchy(
//    /*next_index=*/
//    [](const typename hp::FECollection<dim> &fe_collection,
//       const unsigned int                    fe_index) -> unsigned int {
//      return ((fe_index + 1) < fe_collection.size()) ? fe_index + 1 : fe_index;
//    },
//    /*previous_index=*/
//    [min_fe_index](const typename hp::FECollection<dim> &,
//                   const unsigned int fe_index) -> unsigned int {
//      Assert(fe_index >= min_fe_index, ExcMessage("Finite element is not part of hierarchy!"));
//      return (fe_index > min_fe_index) ? fe_index - 1 : fe_index;
//    });
//
//  // choose adaptation strategy
//  adaptation_strategy = Factory::create_adaptation<dim, typename LinearAlgebra::Vector, spacedim>(
//    prm.adaptation_type,
//    prm.prm_adaptation,
//    locally_relevant_solution,
//    fe_collection,
//    dof_handler,
//    triangulation);
//
//  // cell weighting
//  if (prm.adaptation_type != "h")
//    {
//      cell_weights.reinit(dof_handler,
//                          parallel::CellWeights<dim, spacedim>::ndofs_weighting(
//                            {prm.prm_adaptation.weighting_factor,
//                             prm.prm_adaptation.weighting_exponent}));
//    }
//}
//
//
//
//template<int dim, typename LinearAlgebra, int spacedim>
//void
//ProblemBase<dim, LinearAlgebra, spacedim>::initialize_grid()
//{
//  TimerOutput::Scope t(getTimer(), "initialize_grid");
//
//  Factory::create_grid(prm.grid_type, triangulation);
//
//  if (prm.resume_filename.compare("") != 0)
//    {
//      resume_from_checkpoint();
//    }
//  else
//    {
//      const unsigned int min_fe_index = prm.prm_adaptation.min_p_degree - 1;
//      for (const auto &cell : dof_handler.active_cell_iterators())
//        if (cell->is_locally_owned())
//          cell->set_active_fe_index(min_fe_index);
//
//      dof_handler.distribute_dofs(fe_collection);
//
//      triangulation.refine_global(adaptation_strategy->get_n_initial_refinements());
//    }
//}
//
//
//
//template <int dim, typename LinearAlgebra, int spacedim>
//void
//ProblemBase<dim, LinearAlgebra, spacedim>::compute_errors()
//{
//  TimerOutput::Scope t(getTimer(), "compute_errors");
//
//  Vector<float> difference_per_cell(triangulation.n_active_cells());
//  VectorTools::integrate_difference(dof_handler,
//                                    locally_relevant_solution,
//                                    *solution_function,
//                                    difference_per_cell,
//                                    quadrature_collection,
//                                    VectorTools::L2_norm);
//  const double L2_error =
//    VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
//
//  VectorTools::integrate_difference(dof_handler,
//                                    locally_relevant_solution,
//                                    *solution_function,
//                                    difference_per_cell,
//                                    quadrature_collection,
//                                    VectorTools::H1_norm);
//  const double H1_error =
//    VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::H1_norm);
//
//  getPCOut() << "   L2 error:                     " << L2_error << std::endl
//             << "   H1 error:                     " << H1_error << std::endl;
//
//  TableHandler &table = getTable();
//  table.add_value("L2", L2_error);
//  table.add_value("H1", H1_error);
//  table.set_scientific("L2", true);
//  table.set_scientific("H1", true);
//}
//
//
//
//template <int dim, typename LinearAlgebra, int spacedim>
//void
//ProblemBase<dim, LinearAlgebra, spacedim>::output_results()
//{
//  TimerOutput::Scope t(getTimer(), "output_results");
//
//  Vector<float> fe_degrees(triangulation.n_active_cells());
//  for (const auto &cell : dof_handler.active_cell_iterators())
//    if (cell->is_locally_owned())
//      fe_degrees(cell->active_cell_index()) = cell->get_fe().degree;
//
//  Vector<float> subdomain(triangulation.n_active_cells());
//  for (auto &subd : subdomain)
//    subd = triangulation.locally_owned_subdomain();
//
//  DataOut<dim> data_out;
//  data_out.attach_dof_handler(dof_handler);
//
//  data_out.add_data_vector(locally_relevant_solution, "solution");
//  data_out.add_data_vector(fe_degrees, "fe_degree");
//  data_out.add_data_vector(subdomain, "subdomain");
//
//  if (adaptation_strategy->get_error_estimates().size() > 0)
//    data_out.add_data_vector(adaptation_strategy->get_error_estimates(), "error");
//  if (adaptation_strategy->get_hp_indicators().size() > 0)
//    data_out.add_data_vector(adaptation_strategy->get_hp_indicators(), "hp_indicator");
//
//  data_out.build_patches(mapping_collection);
//
//  data_out.write_vtu_with_pvtu_record("./", prm.file_stem, cycle, mpi_communicator, 2, 1);
//}
//
//
//
//template<int dim, typename LinearAlgebra, int spacedim>
//void
//ProblemBase<dim, LinearAlgebra, spacedim>::resume_from_checkpoint()
//{
//  triangulation.load(prm.resume_filename);
//
//  // custom repartitioning using DoFs requires correctly assigned FEs
//  dof_handler.deserialize_active_fe_indices();
//  dof_handler.distribute_dofs(fe_collection);
//  triangulation.repartition();
//
//  // unpack after repartitioning to avoid unnecessary data transfer
//  adaptation_strategy->unpack_after_serialization();
//
//  // load metadata
//  std::ifstream                   file(prm.resume_filename + ".metadata", std::ios::binary);
//  boost::archive::binary_iarchive ia(file);
//  ia >> cycle;
//}
//
//
//
//template<int dim, typename LinearAlgebra, int spacedim>
//void
//ProblemBase<dim, LinearAlgebra, spacedim>::write_to_checkpoint()
//{
//  // write triangulation and data
//  dof_handler.prepare_for_serialization_of_active_fe_indices();
//  adaptation_strategy->prepare_for_serialization();
//
//  const std::string filename = prm.file_stem + "-checkpoint";
//  triangulation.save(filename);
//
//  // write metadata
//  std::ofstream                   file(filename + ".metadata", std::ios::binary);
//  boost::archive::binary_oarchive oa(file);
//  oa << cycle;
//
//  getPCOut() << "Checkpoint written." << std::endl;
//}
//
//
//
//template <int dim, typename LinearAlgebra, int spacedim>
//void
//ProblemBase<dim, LinearAlgebra, spacedim>::run()
//{
// getTable().set_auto_fill_mode(true);
//
// for (cycle = 0; cycle < adaptation_strategy->get_n_cycles(); ++cycle)
//   {
//     {
//       TimerOutput::Scope t(getTimer(), "full_cycle");
//
//       if (cycle == 0)
//         {
//           initialize_grid();
//         }
//       else
//         {
//           adaptation_strategy->refine();
//
//           if ((prm.checkpoint_frequency > 0) && (cycle % prm.checkpoint_frequency == 0))
//             write_to_checkpoint();
//         }
//
//       Log::log_cycle(cycle, prm);
//
//       setup_system();
//
//       solve();
//
//       compute_errors();
//       adaptation_strategy->estimate_mark();
//
//       if ((prm.output_frequency > 0) && (cycle % prm.output_frequency == 0))
//         output_results();
//     }
//
//     Log::log_timings();
//
//     if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
//       {
//         std::ofstream logstream(filename_log);
//         getTable().write_text(logstream);
//       }
//
//     getTimer().reset();
//     getTable().start_new_row();
//   }
//}
