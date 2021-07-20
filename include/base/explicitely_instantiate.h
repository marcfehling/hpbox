// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
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

#ifndef base_explicitely_instantiate_h
#define base_explicitely_instantiate_h


#if defined(DEAL_II_WITH_TRILINOS) && defined(DEAL_II_WITH_PETSC)
#  define EXPLICITLY_INSTANTIATE(T)         \
    template class T<2, dealiiTrilinos, 2>; \
    template class T<3, dealiiTrilinos, 3>; \
    template class T<2, Trilinos, 2>;       \
    template class T<3, Trilinos, 3>;       \
    template class T<2, PETSc, 2>;          \
    template class T<3, PETSc, 3>;
#elif defined(DEAL_II_WITH_TRILINOS)
#  define EXPLICITLY_INSTANTIATE(T)         \
    template class T<2, dealiiTrilinos, 2>; \
    template class T<3, dealiiTrilinos, 3>; \
    template class T<2, Trilinos, 2>;       \
    template class T<3, Trilinos, 3>;
#elif defined(DEAL_II_WITH_PETSC)
#  define EXPLICITLY_INSTANTIATE(T) \
    template class T<2, PETSc, 2>;  \
    template class T<3, PETSc, 3>;
#endif


#endif
