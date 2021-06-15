## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the deal.II authors
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
# parallel-matrixfree-hp
#

CMAKE_MINIMUM_REQUIRED( VERSION 3.1.0 )



#
# Find deal.II
#
FIND_PACKAGE( deal.II 10.0.0 QUIET
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
IF( NOT DEAL_II_WITH_P4EST OR NOT DEAL_II_WITH_TRILINOS )
  MESSAGE( FATAL_ERROR "
Error! This program requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_P4EST = ON
    DEAL_II_WITH_TRILINOS = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options
    DEAL_II_WITH_P4EST = ${DEAL_II_WITH_P4EST}
    DEAL_II_WITH_TRILINOS = ${DEAL_II_WITH_TRILINOS}
which conflict with the requirements."
    )
ENDIF()



#
# Setup project
#

# Set CXX requirements
SET( CXX_STANDARD 17 )
SET( CXX_STANDARD_REQUIRED ON )

# Initialize project
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT( "parallel-matrixfree-hp" )

INCLUDE_DIRECTORIES( include )

# Declare all files of the library
FILE( GLOB_RECURSE LIB_SRC "${CMAKE_SOURCE_DIR}/source/*.cc" )
FILE( GLOB_RECURSE LIB_INC "${CMAKE_SOURCE_DIR}/include/*.h" )
ADD_LIBRARY( pmfhp ${LIB_SRC} ${LIB_INC} )
DEAL_II_SETUP_TARGET( pmfhp )

# Setup individual targets
FILE( GLOB_RECURSE APP_SRC "${CMAKE_SOURCE_DIR}/application/*.cc" )
FOREACH( APP_FILE ${APP_SRC})
  # Extract filename without extension
  GET_FILENAME_COMPONENT( APP_NAME ${APP_FILE} NAME_WE )
  ADD_EXECUTABLE( ${APP_NAME} ${APP_FILE} )
  # Add libraries
  DEAL_II_SETUP_TARGET( ${APP_NAME} )
  TARGET_LINK_LIBRARIES( ${APP_NAME} pmfhp )
ENDFOREACH( APP_FILE ${APP_SRC} )
