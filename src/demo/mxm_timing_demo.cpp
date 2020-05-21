/*
 * GraphBLAS Template Library, Version 2.1
 *
 * Copyright 2019 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors. All Rights Reserved.
 *
 * THIS MATERIAL WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN AGENCY OF
 * THE UNITED STATES GOVERNMENT.  NEITHER THE UNITED STATES GOVERNMENT NOR THE
 * UNITED STATES DEPARTMENT OF ENERGY, NOR THE UNITED STATES DEPARTMENT OF
 * DEFENSE, NOR CARNEGIE MELLON UNIVERSITY, NOR BATTELLE, NOR ANY OF THEIR
 * EMPLOYEES, NOR ANY JURISDICTION OR ORGANIZATION THAT HAS COOPERATED IN THE
 * DEVELOPMENT OF THESE MATERIALS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
 * OR USEFULNESS OR ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS
 * DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED
 * RIGHTS..
 *
 * Released under a BSD (SEI)-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * DM18-0559
 */

#include <iostream>
#include <fstream>
#include <chrono>

#define GRAPHBLAS_DEBUG 1

#include <graphblas/graphblas.hpp>
#include <algorithms/triangle_count.hpp>
#include "Timer.hpp"

using namespace GraphBLAS;

//****************************************************************************
IndexType read_edge_list(std::string const &pathname,
                         IndexArrayType &row_indices,
                         IndexArrayType &col_indices)
{
    std::ifstream infile(pathname);
    IndexType max_id = 0;
    uint64_t num_rows = 0;
    uint64_t src, dst;

    while (infile)
    {
        infile >> src >> dst;
        //std::cout << "Read: " << src << ", " << dst << std::endl;
        max_id = std::max(max_id, src);
        max_id = std::max(max_id, dst);

        //if (src > max_id) max_id = src;
        //if (dst > max_id) max_id = dst;

        row_indices.push_back(src);
        col_indices.push_back(dst);

        ++num_rows;
    }
    std::cout << "Read " << num_rows << " rows." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    return (max_id + 1);
}


//****************************************************************************
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments." << std::endl;
        std::cerr << "Usage: " << argv[0] << " <edge list file>" << std::endl;
        exit(1);
    }

    // Read the edgelist and create the tuple arrays
    std::string pathname(argv[1]);
    IndexArrayType iA, jA;

    IndexType const NUM_NODES(read_edge_list(pathname, iA, jA));

    typedef int32_t T;
    typedef Matrix<T> MatType;
    typedef Matrix<bool> BoolMatType;
    std::vector<T> v(iA.size(), 1);
    std::vector<bool> bv(iA.size(), true);
    MatType A(NUM_NODES, NUM_NODES);
    MatType B(NUM_NODES, NUM_NODES);
    BoolMatType M(NUM_NODES, NUM_NODES);

    A.build(iA.begin(), jA.begin(), v.begin(), iA.size());
    B.build(iA.begin(), jA.begin(), v.begin(), iA.size());
    M.build(iA.begin(), jA.begin(), bv.begin(), iA.size());

    std::cout << "Running algorithm(s)... nvals = " << M.nvals() << std::endl;
    T count(0);

    Timer<std::chrono::system_clock, std::chrono::microseconds> my_timer;
    MatType C(NUM_NODES, NUM_NODES);
    mxm(C,
        NoMask(),
        NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);

    //=====================================================
    // Perform matrix multiplies with 4 different kernels
    //=====================================================
    //===================
    // Perform matrix multiplies with reference implementation
    std::cout << "OPTIMIZED IMPLEMENTATION:" << std::endl;
    // C.clear();
    // my_timer.start();
    // mxm_original(C, NoMask(), NoAccumulate(),
    //                         ArithmeticSemiring<double>(),
    //                         A, B);
    // my_timer.stop();
    // std::cout << "C := A+.*B                : " << my_timer.elapsed() << " usec, C.nvals = " << C.nvals() << std::endl;

    //===================
    // A*B
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A*B" << std::endl;
    C.clear();
    my_timer.start();
    mxm(C, NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C := A+.*B                : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C := C + A+.*B            : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<M,merge> := A+.*B       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := A+.*B     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<M,merge> := C + A+.*B   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := C + A+.*B : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<!M,merge> := A+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := A+.*B    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<!M,merge> := C + A+.*B  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := C + A+.*B: " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //-------------------- structure only

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<s(M),merge> := A+.*B    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := A+.*B  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<s(M),merge> := C + A+.*B   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := C + A+.*B : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<!s(M),merge> := A+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := A+.*B    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B);
    my_timer.stop();
    std::cout << "C<!s(M),merge> := C + A+.*B  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, B, REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := C + A+.*B: " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //===================
    // A'*B
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A'*B" << std::endl;
    C.clear();
    my_timer.start();
    mxm(C, NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C := A'+.*B                : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C := C + A'+.*B            : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<M,merge> := A'+.*B       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := A+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<M,merge> := C + A'+.*B   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := C + A'+.*B : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<!M,merge> := A'+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := A'+.*B    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<!M,merge> := C + A'+.*B  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := C + A'+.*B: " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //-------------------- structure only

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<s(M),merge> := A'+.*B       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := A+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<s(M),merge> := C + A'+.*B   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := C + A'+.*B : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<!s(M),merge> := A'+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := A'+.*B    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B);
    my_timer.stop();
    std::cout << "C<!s(M),merge> := C + A'+.*B  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), B, REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := C + A'+.*B: " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //===================
    // A*B'
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A*B'" << std::endl;
    C.clear();
    my_timer.start();
    mxm(C, NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C := A+.*B'                : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C := C + A+.*B'            : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<M,merge> := A+.*B'       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := A+.*B'     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<M,merge> := C + A+.*B'   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := C + A+.*B' : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<!M,merge> := A+.*B'      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := A+.*B'    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<!M,merge> := C + A+.*B'  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := C + A+.*B': " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //-------------------- structure only

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<s(M),merge> := A+.*B'       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := A+.*B'     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<s(M),merge> := C + A+.*B'   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := C + A+.*B' : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<!s(M),merge> := A+.*B'      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := A+.*B'    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B));
    my_timer.stop();
    std::cout << "C<!s(M),merge> := C + A+.*B'  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := C + A+.*B': " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //===================
    // A'*B'
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A'*B'" << std::endl;
    C.clear();
    my_timer.start();
    mxm(C, NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C := A+.*B'                : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C := C + A+.*B'            : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<M,merge> := A+.*B'       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := A+.*B'     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<M,merge> := C + A+.*B'   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := C + A+.*B' : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<!M,merge> := A+.*B'      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := A+.*B'    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<!M,merge> := C + A+.*B'  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := C + A+.*B': " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //-------------------- structure only

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<s(M),merge> := A+.*B'       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := A+.*B'     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<s(M),merge> := C + A+.*B'   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, structure(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<s(M),replace> := C + A+.*B' : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<!s(M),merge> := A+.*B'      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := A+.*B'    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B));
    my_timer.stop();
    std::cout << "C<!s(M),merge> := C + A+.*B'  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    mxm(C, complement(structure(M)), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), transpose(B), REPLACE);
    my_timer.stop();
    std::cout << "C<!s(M),replace> := C + A+.*B': " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;


#if 0
    // The follow requires access to the internals of frontend matrix

    //===================
    // Perform matrix multiplies with reference implementation
    std::cout << "REFERENCE IMPLEMENTATION:" << std::endl;

    //===================
    // A*B
    //===================
    std::cout << "REFERENCE IMPLEMENTATION: A*B" << std::endl;
    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        backend::NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A.m_mat, B.m_mat, false);
    my_timer.stop();
    std::cout << "C := A+.*B                : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        backend::NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        A.m_mat, B.m_mat, false);
    my_timer.stop();
    std::cout << "C := C + A+.*B            : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        M.m_mat, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A.m_mat, B.m_mat, false);
    my_timer.stop();
    std::cout << "C<M,merge> := A+.*B       : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        M.m_mat, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A.m_mat, B.m_mat, REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := A+.*B     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        M.m_mat, Plus<double>(),
        ArithmeticSemiring<double>(),
        A.m_mat, B.m_mat, false);
    my_timer.stop();
    std::cout << "C<M,merge> := C + A+.*B   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        M.m_mat, Plus<double>(),
        ArithmeticSemiring<double>(),
        A.m_mat, B.m_mat, REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := A+.*B     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        backend::MatrixComplementView<BoolMatType::BackendType>(M.m_mat),
        NoAccumulate(),
        ArithmeticSemiring<double>(),
        A.m_mat, B.m_mat, false);
    my_timer.stop();
    std::cout << "C<!M,merge> := A+.*B      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        backend::MatrixComplementView<BoolMatType::BackendType>(M.m_mat),
        NoAccumulate(),
        ArithmeticSemiring<double>(),
        A.m_mat, B.m_mat, REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := A+.*B    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        backend::MatrixComplementView<BoolMatType::BackendType>(M.m_mat),
        Plus<double>(),
        ArithmeticSemiring<double>(),
        A.m_mat, B.m_mat, false);
    my_timer.stop();
    std::cout << "C<!M,merge> := C + A+.*B  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        backend::MatrixComplementView<BoolMatType::BackendType>(M.m_mat),
        Plus<double>(),
        ArithmeticSemiring<double>(),
        A.m_mat, B.m_mat, REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := C + A+.*B: " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    //===================
    // A*B'
    //===================
    std::cout << "REFERENCE IMPLEMENTATION: A*B'" << std::endl;
    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        backend::NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A.m_mat,
        backend::TransposeView<MatType::BackendType>(B.m_mat),
        false);
    my_timer.stop();
    std::cout << "C := A+.*B'               : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        backend::NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        A.m_mat,
        backend::TransposeView<MatType::BackendType>(B.m_mat),
        false);
    my_timer.stop();
    std::cout << "C := C + A+.*B'           : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        M.m_mat, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A.m_mat,
        backend::TransposeView<MatType::BackendType>(B.m_mat),
        false);
    my_timer.stop();
    std::cout << "C<M,merge> := A+.*B'      : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        M.m_mat, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A.m_mat,
        backend::TransposeView<MatType::BackendType>(B.m_mat),
        REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := A+.*B'    : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        M.m_mat, Plus<double>(),
        ArithmeticSemiring<double>(),
        A.m_mat,
        backend::TransposeView<MatType::BackendType>(B.m_mat),
        false);
    my_timer.stop();
    std::cout << "C<M,merge> := C + A+.*B'  : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        M.m_mat, Plus<double>(),
        ArithmeticSemiring<double>(),
        A.m_mat,
        backend::TransposeView<MatType::BackendType>(B.m_mat),
        REPLACE);
    my_timer.stop();
    std::cout << "C<M,replace> := C + A+.*B': " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        backend::MatrixComplementView<BoolMatType::BackendType>(M.m_mat),
        NoAccumulate(),
        ArithmeticSemiring<double>(),
        A.m_mat,
        backend::TransposeView<MatType::BackendType>(B.m_mat),
        false);
    my_timer.stop();
    std::cout << "C<!M,merge> := A+.*B'     : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        backend::MatrixComplementView<BoolMatType::BackendType>(M.m_mat),
        NoAccumulate(),
        ArithmeticSemiring<double>(),
        A.m_mat,
        backend::TransposeView<MatType::BackendType>(B.m_mat),
        REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := A+.*B'   : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;


    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        backend::MatrixComplementView<BoolMatType::BackendType>(M.m_mat),
        Plus<double>(),
        ArithmeticSemiring<double>(),
        A.m_mat,
        backend::TransposeView<MatType::BackendType>(B.m_mat),
        false);
    my_timer.stop();
    std::cout << "C<!M,merge> := C + A+.*B' : " << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;

    my_timer.start();
    backend::original_mxm(
        C.m_mat,
        backend::MatrixComplementView<BoolMatType::BackendType>(M.m_mat),
        Plus<double>(),
        ArithmeticSemiring<double>(),
        A.m_mat,
        backend::TransposeView<MatType::BackendType>(B.m_mat),
        REPLACE);
    my_timer.stop();
    std::cout << "C<!M,replace> := C + A+.*B':" << my_timer.elapsed()
              << " usec, C.nvals = " << C.nvals() << std::endl;
#endif

    return 0;
}
