# Metallizing GraphBlas Template Library

The goal of this project is to prototype a persistent memory allocator for GBTL container.

Metall is a memory allocator for persistent memory. Version 0.6
More Info at : https://github.com/LLNL/metall/releases/tag/v0.6

GraphBlas Template Library is a modern idiomatic C++ reference implementation of the GraphBLAS C API Specification and has examples of commonly used graph algorithms implemented with the GraphBLAS primitive operations.
More Info at: https://github.com/cmu-sei/gbtl. This repository uses the GBTL master branch.

## Summary of the changes:

#### 1. gbtl/src/graphblas/demo/triangle_count_demo.cpp

The `grb::Matrix` type is now a metall matrix type (persistent type) with metall manager allocator.
There are 2 scopes in the application program. 
In the first scope, we create a metall manager and construct a metall matrix L. We then build the L matrix as usual. 
In the second scope, we reattach to the previously created metall manager and call the algorithm on the metall matrix L.
Here, only `algorithms::triangle_count_masked(*L)` is used. `(C<L> = L +.* L'; #=|C|)`


#### 2. gbtl/src/algorithms/triangle_count.hpp

No changes here. Just commented out all the other triangle counting algorithms except `|L.*(L +.* L')| triangle_count_masked(L)`. Note that, the temporary B matrix uses `MatrixT::ScalarType` (Non-persistent type) and not metall matrix (persistent type)


#### 3. gbtl/src/graphblas/Matrix.hpp

Removed all tags in frontend matrix class `matrix_generator BackendType` and  uses direct ` using BackendType = grb::backend::LilSparseMatrix<ScalarT, allocator_t>;`
In the template parameters, we added `typename allocator_t = std::allocator<char>` as an additional argument to the matrix class. This allocator_t is passed down to all the constructors in this Matrix class and wherever template parameters are used.

                
#### 4. gbtl/src/graphblas/types.hpp

Added a template parameter to the frontend Matrix class wrapper `typename Metall_Manager_Alloc_Type`


#### 5. gbtl/src/graphblas/platforms/metall-gbtl-platform/LilSparseMatrix.hpp

Vector of vectors with scoped allocator adaptor is added here. In multi-level containers, you have to use scoped_allocator_adaptor in the most outer container (`outer_vector_type`) so that the inner containers (`RowType` or the inner vector) obtain their allocator arguments from the outer containers's scoped_allocator_adaptor. The `element type` is bound to std::allocator_traits.

In the template parameters, we added an additional argument `typename allocator_t = std::allocator<char>` to the `LilSparseMatrix` class. This `allocator_t` is passed down to all the constructors in this `LilSparseMatrix` class and whereever template parameters are used. This allocator is assigned to `m_data`. 



#### 6. gbtl/src/graphblas/platforms/metall-gbtl-platform/sparse_helpers.hpp

In the `dot` function, `reduction` function, and `apply_with_mask` function, We added a argument `typename allocator_t` in the template parameters and replaced `std::vector` with `boost::container::vector`.

Hardcoded point fix at `apply_with_mask` function (should be replaced in future)
    
    result.emplace_back(mask_idx, static_cast<CScalarT>(std::get<1>(*z_it)));
    result.emplace_back(mask_idx, static_cast<int>(std::get<1>(*z_it)));




### Summary

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

    g++   -std=gnu++1z                                          # Requires gcc9
          -I./src/graphblas/detail                              # Include header files from gbtl detail dir
          -I./src                                               # Include header files from gbtl src dir
          -I./src/graphblas/platforms/metall-gbtl-platform      # Include header files from gbtl metall-gbtl-platform dir
          -I./metall/include/                               # Include header files from metall include dir
          -I/path/to/boost/include                              # Include header files from boost include dir [mostly optional]
          -L/usr/lib/gcc/lib64                                  # Link with the gcc library directory [mostly optional]
          ./src/demo/triangle_count_demo.cpp                    # My cpp program
          -o                                                    # Specify my build output file
          gbtl_tc.exe                                           # My executable file
          -lstdc++fs                                            # Required by metall to use the Filesystem library



          g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/  ./src/demo/triangle_count_demo.cpp   -o  gbtl_tc.exe   -lstdc++fs   

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
        -I./metall/include/
        ./metall/example/adjacency_list_graph.cpp
        -o
        adjacency_list_graph.exe  
        -lstdc++fs


        g++ -std=c++17 -I./metall/include/ ./metall/example/adjacency_list_graph.cpp -o  adjacency_list_graph.exe -lstdc++fs

### To Run Just Metall:

    ./adjacency_list_graph.exe




  

## Results

  

Just using algorithms::triangle_count_masked(L)

  

### With sequential platform backend type

    [velusamy@flash21:gbtl]$ g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/sequential src/demo/triangle_count_demo.cpp -o gbtl_tc                             
    [velusamy@flash21:gbtl]$ ./gbtl_tc src/demo/triangle_count_data_ca-HepTh.tsv 
    Elapsed read time: 15539 usec.
    Read 51947 rows.
    #Nodes = 9877
    Elapsed sort/relabel time: 23993 usec.
    0 <-- 86: deg = 65
    1 <-- 15: deg = 60
    2 <-- 54: deg = 59
    ...
    9874 <-- 4561: deg = 1
    9875 <-- 9549: deg = 0
    9876 <-- 9600: deg = 0
    Running algorithm(s)...
    ...
    # triangles (C<L> = L +.* L'; #=|C|) = 28339
    Elapsed time: 4.31759e+07 usec.
    ...

### With metall-gbtl-platform platform backend type

    [velusamy@flash21:gbtl]$ g++ -std=gnu++1z -I./src/graphblas/detail -I./src -I./src/graphblas/platforms/metall-gbtl-platform  -I./metall/include/  ./src/demo/triangle_count_demo.cpp   -o  gbtl_tc.exe   -lstdc++fs
    [velusamy@flash21:gbtl]$ ./gbtl_tc.exe src/demo/triangle_count_data_ca-HepTh.tsv 
    Elapsed read time: 18941 usec.
    Read 51947 rows.
    #Nodes = 9877
    Elapsed sort/relabel time: 24142 usec.
    0 <-- 86: deg = 65
    1 <-- 15: deg = 60
    2 <-- 54: deg = 59
    ....
    9875 <-- 9549: deg = 0
    9876 <-- 9600: deg = 0
    Running algorithm(s)...
    # triangles (C<L> = L +.* L'; #=|C|) = 28339
    Elapsed time: 1.15253e+08 usec.
    Running algorithm(s)...
    # triangles (C<L> = L +.* L; #=|C|) = 28339
    Elapsed time: 111927 usec.




