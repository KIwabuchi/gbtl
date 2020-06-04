# GraphBLAS Template Library (GBTL), v. 3.0

## Project Goals

* Complete, mathematically equivalent implementation of the GraphBLAS
C API Specification (current specification http://graphblas.org).
* Use of modern, idiomatic C++ for the implementation and provide a
testing ground for the GraphBLAS C++ API Specification development.
* Graph algorithm library containing examples of commonly used graph
algorithms implemented with the GraphBLAS primitive operations.

This is Version 3.0 of the C++ implementation and is mathematically
equivalent to Version 1.3 of the GraphBLAS C API.  The API defined by
this (v. 3.0) release is not backward compatible with GBTL v. 2.0. See
the ChangeLog file for details.

The project also contains example implementations of many common graph
algorithms using the C++ API:

* Breadth-first traversal (aka BFS)
  * level BFS
  * parent list BFS
  * batched BFS
* Single-source shortest path (SSSP)
  * Bellman-Ford
  * Filtered Bellman-Ford
  * Delta stepping
* All-pairs shortest path (APSP)
* Centrality measures
  * Vertex Betweenness Centrality (batch variant too)
  * Edge Betweenness Centrality
  * Closeness centrality
* Clustering
  * peer pressure clustering
  * Markov clustering
  * Louvain
* Triangle counting (many variants)
* K-truss enumeration
  * incidence matrix variant
  * adjacency matrix variant
* PageRank
* Maximal Independent Set (MIS)
* Minimum Spanning Tree (MST)
* Maxflow
  * Ford-Fulkerson
* Metrics
  * degree, in and out
  * graph distance
  * radius, diameter
  * vertex eccentricity

Work is underway to port some of the algorithms in LAGraph repository to GBTL.

## Backend Implementations Available

The file structure and build system support defining multiple
different 'backend' implementations (platforms), but only one of these
can be configured and compiled at a time.  However, if multiple
different build directories are used (extending what is shown below),
then each can configured for a different platform.

This release contains the 'sequential' backend in the platforms
directory that is written for a single CPU. It is intended as a
reference implementation focusing on correctness, but contains some
modest performance improvements over previous releases.  An
experimental platform called 'optimized_sequential' is currently under
development and is exploring more comprehensive performance
improvements (currently only for the mxm operation).

Support for GPUs that was in version 1.0 is currently not available
but can be accessed using the git tag: '1.0.0').

## Building

### Prerequisites

A detailed study of which C++ compilers are required has not been
carried out.  The cmake build system is currently configured to
require C++17 support.  I use a g++-9 compiler.

Building the unit tests also requires the "Boost Test Library: The
Unit Test Framework."

### Compilation

This project is designed to use cmake to build and use an "out of
source" style build to make it easy to clean up. The tests and demos can be
built in to a "build" directory in the top-level directory by following
these steps:

```
$ mkdir build
$ cd build
$ cmake [-DPLATFORM=<platform name>] [-DCMAKE_BUILD_TYPE={Release,Debug}] ../src
$ make
```

The optional `PLATFORM` argument to `cmake` specifies which platform-specific
source code (also referred to as the backend) should be configured for the
build and the value must correspond to a subdirectory in
"gbtl/src/graphblas/platforms/" and that subdirectory must have a
"backend_include.hpp" file.  If this argument is omitted it defaults to
configuring the "sequential" platform. The other platform currently available
is "optimized_sequential" which is currently under development to improve the
performance of various operations.

The optional `CMAKE_BUILD_TYPE` argument to `cmake` can be used to build debug
or release (using `-O3` compiler option) versions of the library. The default is
`Debug`.

The compiler used to build the library can be changed by
specifying `-DCXX=<pathname_to_compiler>` on the cmake commandline as well.

Once cmake is done building the Makefiles, options to make can be used.  For
example, using "make -i -j8" tries to build every test (ignoring all errors)
and uses 8 threads to speed up the build (use a number appropriate for the
number of cores/hyperthreads on your system).

There is a convenience script to do all of this from scratch called
rebuild.sh that also removes all the old content from a previous build.

For CLion support in the cmake project settings "Build, Execution,
Deployment > CMake > Generation path:" set it to "../build" to use the
same makefiles as that created by the clean build process so that
there aren't two different build trees.

### Installation

The current library is set up as a header only library.  To install this
library, copy the graphblas directory, its subdirectories and the
specific platform subdirectory (sans the platform's test directories) to
a location in your include path.

### Documentation

Documentation can be generated using the Doxygen documentation system.  To
generate documentation run doxygen from the src directory:

```
$ cd src
$ doxygen
```

All documentation is built in the 'docs' subdirectory.

## Acknowledgments and Disclaimers

This material is based upon work funded and supported by the United
States Department of Defense under Contract No. FA8702-15-D-0002 with
Carnegie Mellon University for the operation of the Software
Engineering Institute, a federally funded research and development
center and by the United States Department of Energy under Contract
DE-AC05-76RL01830 with Battelle Memorial Institute for the Operation
of the Pacific Northwest National Laboratory.

THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN
AGENCY OF THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES
GOVERNMENT NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED
STATES DEPARTMENT OF DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR
BATTELLE, NOR ANY OF THEIR EMPLOYEES, NOR ANY JURISDICTION OR
ORGANIZATION THAT HAS COOPERATED IN THE DEVELOPMENT OF THESE
MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY
LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR
USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR
PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE
PRIVATELY OWNED RIGHTS.

Reference herein to any specific commercial product, process, or
service by trade name, trademark, manufacturer, or otherwise does not
necessarily constitute or imply its endorsement, recommendation, or
favoring by the United States Government or any agency thereof, or
Carnegie Mellon University, or Battelle Memorial Institute. The views
and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

DM20-0442

Please see “AUTHORS” file for a list of known contributors.

This release is an update of:

1. GraphBLAS Template Library (GBTL)
(https://github.com/cmu-sei/gbtl/blob/1.0.0/LICENSE) Copyright 2015
Carnegie Mellon University and The Trustees of Indiana. DM17-0037,
DM-0002659

2. GraphBLAS Template Library (GBTL)
(https://github.com/cmu-sei/gbtl/blob/2.0.0/LICENSE) Copyright 2018
Carnegie Mellon University, Battelle Memorial Institute, and Authors.
DM18-0559
