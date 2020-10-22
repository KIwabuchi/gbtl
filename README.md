# Metallizing GraphBlas Template Library

The goal of this project is to prototype a persistent memory allocator for GBTL container.

Metall is a memory allocator for persistent memory. Version 0.5
More Info at : https://github.com/LLNL/metall/releases/tag/v0.5

GraphBlas Template Library is a modern idiomatic C++ reference implementation of the GraphBLAS C API Specification and has examples of commonly used graph algorithms implemented with the GraphBLAS primitive operations.
More Info at: https://github.com/cmu-sei/gbtl


This repository uses the GBTL master branch and has 4 major changes.

1. Removed all tags in frontend matrix class "matrix_generator" BackendType and  uses
      `BackendType = grb::backend::LilSparseMatrix<ScalarT>;`
2. Added multilevel containers with metall scoped allocator to "LilSparseMatrix"  ElementType, RowType, outer_vector_type m_data with` boost::container::vector` instead of `std::vector`
3. Added `" template<typename ScalarT, typename allocator_t = std::allocator<char>, typename... TagsT>"` to the frontend Matrix constructors and the backend LilSparseMatrix constructors.
4. Hardcoded template parameters at  sparse_helpers.hpp        dot(); reduction(); apply_with_mask(); with` boost::container::vector<std::tuple<grb::IndexType,D1>, allocator_t > `instead of `std::vector<std::tuple<grb::IndexType,D1> >`



## Getting Started


### Requirements:

    cd gbtl
    git checkout metall-gbtl

    module load gcc/8.1.0

    spack install metall
    spack install gcc@9.3.0
    spack install boost

    spack load metall
    spack load gcc@9.3.0
    spack load boost

### To Compile GBTL with Metall:

    g++   -std=gnu++1z                               # Requires gcc9
          -I./src/graphblas/detail                   # Include header files from gbtl detail dir
          -I./src                                    # Include header files from gbtl src dir
          -I./src/graphblas/platforms/sequential     # Include header files from gbtl sequential-platform dir
          -I./metall-0.5/include/                    # Include header files from metall include dir
          -I/path/to/boost/include                   # Include header files from boost include dir [mostly optional]
          -L/usr/lib/gcc/lib64                       # Link with the gcc library directory [mostly optional]
          ./src/demo/triangle_count_demo.cpp         # My cpp program
          -o                                         # Specify my build output file
          gbtl_tc.exe                                # My executable file
          -lstdc++fs                                 # Required by metall to use the Filesystem library



          g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/sequential  -I./metall-0.5/include/  ./src/demo/triangle_count_demo.cpp   -o  gbtl_tc.exe   -lstdc++fs   

### To Run:

    ./gbtl_tc.exe ./src/demo/triangle_count_data_ca-HepTh.tsv


### To Compile just GBTL (Triangle Counting):

    g++ -std=gnu++1z
          -I./src/graphblas/detail
          -I./src
          -I./src/graphblas/platforms/sequential
          src/demo/triangle_count_demo.cpp
          -o
          gbtl_tc


### To Run just GBTL (Triangle Counting):

    ./gbtl_tc ./src/demo/triangle_count_data_ca-HepTh.tsv



### To Compile Just Metall:

    g++ -std=c++17
        -I/path/to/boost/include
        -L/usr/lib/gcc/lib64
        -I./metall-0.5/include/
        ./metall-0.5/example/adjacency_list_graph.cpp
        -o
        adjacency_list_graph.exe  
        -lstdc++fs


        g++ -std=c++17 -I./metall-0.5/include/ ./metall-0.5/example/adjacency_list_graph.cpp -o  adjacency_list_graph.exe -lstdc++fs

### To Run Just Metall:

    ./adjacency_list_graph.exe
