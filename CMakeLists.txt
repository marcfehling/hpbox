## ---------------------------------------------------------------------
##
## Copyright (C) 2020 - 2023 by the deal.II authors
##
## This file is part of the deal.II library.
##
## The deal.II library is free software; you can use it, redistribute
## it, and/or modify it under the terms of the GNU Lesser General
## Public License as published by the Free Software Foundation; either
## version 2.1 of the License, or (at your option) any later version.
## The full text of the license can be found in the file LICENSE.md at
## the top level directory of deal.II.
##
## ---------------------------------------------------------------------

#
# hpbox
#

CMAKE_MINIMUM_REQUIRED( VERSION 3.13.4 )



#
# Find deal.II
#
FIND_PACKAGE( deal.II 9.5.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
  )
IF( NOT ${deal.II_FOUND} )
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()



#
# Are all dependencies fulfilled?
#
IF( NOT DEAL_II_WITH_P4EST OR NOT DEAL_II_WITH_LAPACK OR NOT ((DEAL_II_WITH_PETSC AND NOT DEAL_II_PETSC_WITH_COMPLEX) OR DEAL_II_WITH_TRILINOS))
  MESSAGE( FATAL_ERROR "
Error! This program requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_P4EST = ON
    DEAL_II_WITH_LAPACK = ON
    DEAL_II_WITH_TRILINOS = ON
or
    DEAL_II_WITH_P4EST = ON
    DEAL_II_WITH_LAPACK = ON
    DEAL_II_WITH_PETSC = ON
    DEAL_II_PETSC_WITH_COMPLEX = OFF
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options
    DEAL_II_WITH_P4EST = ${DEAL_II_WITH_P4EST}
    DEAL_II_WITH_LAPACK = ${DEAL_II_WITH_LAPACK}
    DEAL_II_WITH_TRILINOS = ${DEAL_II_WITH_TRILINOS}
    DEAL_II_WITH_PETSC = ${DEAL_II_WITH_PETSC}
    DEAL_II_PETSC_WITH_COMPLEX = ${DEAL_II_PETSC_WITH_COMPLEX}
which conflict with the requirements."
    )
ENDIF()



#
# Set up project
#

# Set CXX requirements
SET( CMAKE_CXX_STANDARD 17 )
SET( CMAKE_CXX_STANDARD_REQUIRED ON )
SET( CMAKE_CXX_EXTENSIONS OFF )

# Initialize project
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT( "hpbox" )

INCLUDE_DIRECTORIES( include )

# Declare all files of the library
FILE( GLOB_RECURSE LIB_SRC "${CMAKE_SOURCE_DIR}/source/*.cc" )
FILE( GLOB_RECURSE LIB_INC "${CMAKE_SOURCE_DIR}/include/*.h" )
ADD_LIBRARY( hpbox ${LIB_SRC} ${LIB_INC} )
DEAL_II_SETUP_TARGET( hpbox )

# Setup individual targets
FILE( GLOB_RECURSE APP_SRC "${CMAKE_SOURCE_DIR}/application/*.cc" )
FOREACH( APP_FILE ${APP_SRC} )
  # Extract filename without extension
  GET_FILENAME_COMPONENT( APP_NAME ${APP_FILE} NAME_WE )
  ADD_EXECUTABLE( ${APP_NAME} ${APP_FILE} )
  # Add libraries
  DEAL_II_SETUP_TARGET( ${APP_NAME} )
  TARGET_LINK_LIBRARIES( ${APP_NAME} hpbox )
ENDFOREACH( APP_FILE ${APP_SRC} )



#
# Set up tests
#

ENABLE_TESTING()

FOREACH( APP_FILE ${APP_SRC} )
  # Extract filename without extension
  GET_FILENAME_COMPONENT( APP_NAME ${APP_FILE} NAME_WE )
  FILE( GLOB_RECURSE TEST_SRC "${CMAKE_SOURCE_DIR}/tests/${APP_NAME}/*.prm" )
  FOREACH( TEST_FILE ${TEST_SRC} )
    GET_FILENAME_COMPONENT( TEST_NAME ${TEST_FILE} NAME_WE )
    SET( MPIRUN_ARGS -np 2 ${APP_NAME} ${TEST_FILE} )
    ADD_TEST( ${APP_NAME}_${TEST_NAME} mpirun ${MPIRUN_ARGS} )
  ENDFOREACH( TEST_FILE ${TEST_SRC} )
ENDFOREACH( APP_FILE ${APP_SRC} )
