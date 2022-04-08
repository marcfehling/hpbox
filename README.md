Sandbox for hp-adaptive methods
===============================

This program is used as a benchmark to demonstrate the benefits of
parallelization, hp-adaptive methods and matrix-free methods combined.

All results of the following paper have been generated with this
program.

	@article{fehling_bangerth_2022,
	  author  = {Fehling, Marc and Bangerth, Wolfgang},
	  title   = {Algorithms for Massively Parallel Generic hp--adaptive Finite Element Software},
	  journal = {?},
	  volume  = {?},
	  issue   = {?},
	  year    = {?},
	  doi     = {?}
	}


Dependencies
------------

This program requires a deal.II library built from the current master
branch (version 9.4.0-pre). It needs to be configured with both p4est
and LAPACK.

Further, your deal.II library has to be configured with either PETSc or
Trilinos for algebraic multigrid (AMG) methods. The geometric multigrid
(GMG) implementation currently requires Trilinos and is not compatible
with PETSc.


Compiling and Running
---------------------

To generate a makefile for this code using CMake, create a build
directory to your liking and type the following command into the
terminal from the build directory:

	cmake /path/to/hpbox -DDEAL_II_DIR=/path/to/deal.II

To compile the program with all of the applications, call:

	make
  
An executable named `hprun` will be created. Running the application
will automatically generate a default parameter file in the folder where
the executable is located, and uses the default parameter set for the
current run.

A selection of parameter files for different scenarios is located in the
`examples` folder.


Acknowledgments
---------------

The author would like to thank
Peter Munch ([@peterrum](https://github.com/peterrum/))
for lots of detailed discussions on the multigrid topic.
