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
#include <random>

#define GRAPHBLAS_DEBUG 1

#include <graphblas/graphblas.hpp>
#include "Timer.hpp"

using namespace GraphBLAS;

//****************************************************************************
IndexType read_edge_list(std::string const &pathname,
                         IndexArrayType    &row_indices,
                         IndexArrayType    &col_indices)
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
    IndexArrayType iA, jA, iu;

    IndexType const NUM_NODES(read_edge_list(pathname, iA, jA));

    typedef int32_t T;
    typedef Matrix<T> MatType;
    typedef Vector<T> VecType;
    typedef Vector<bool> BoolVecType;
    std::vector<T> v(iA.size(), 1);
    std::vector<bool> bv(iA.size(), true);
    MatType A(NUM_NODES, NUM_NODES);
    VecType u(NUM_NODES);
    VecType w(NUM_NODES);
    BoolVecType M(NUM_NODES);

    A.build(iA.begin(), jA.begin(), v.begin(), iA.size());

    std::default_random_engine  generator;
    std::uniform_real_distribution<double> distribution;
    for (IndexType iu = 0; iu < NUM_NODES; ++iu)
    {
        if (distribution(generator) < 0.15)
            M.setElement(iu, true);
        if (distribution(generator) < 0.1)
            u.setElement(iu, 1);
    }

    std::cout << "Running algorithm(s)... M.nvals = " << M.nvals() << std::endl;
    std::cout << "u.nvals = " << u.nvals() << std::endl;
    T count(0);

    Timer<std::chrono::system_clock> my_timer;

    // warm up
    mxv(w, NoMask(), NoAccumulate(), ArithmeticSemiring<double>(), A, u);

    //=====================================================
    // Perform matrix vector multiplies
    //=====================================================

    //===================
    // A*u
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A*u" << std::endl;
    w.clear();
    my_timer.start();
    mxv(w, NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, u);
    my_timer.stop();
    std::cout << "w := A+.*u                : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, u);
    my_timer.stop();
    std::cout << "w := w + A+.*u            : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, u);
    my_timer.stop();
    std::cout << "w<M,merge> := A+.*u       : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, u, REPLACE);
    my_timer.stop();
    std::cout << "w<M,replace> := A+.*u     : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, u);
    my_timer.stop();
    std::cout << "w<s(M),merge> := A+.*u    : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, structure(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, u, REPLACE);
    my_timer.stop();
    std::cout << "w<s(M),replace> := A+.*u  : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        A, u);
    my_timer.stop();
    std::cout << "w<M,merge> := w + A+.*u   : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        A, u, REPLACE);
    my_timer.stop();
    std::cout << "w<M,replace> := w + A+.*u : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, u);
    my_timer.stop();
    std::cout << "w<!M,merge> := A+.*u      : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        A, u, REPLACE);
    my_timer.stop();
    std::cout << "w<!M,replace> := A+.*u    : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, u);
    my_timer.stop();
    std::cout << "w<!M,merge> := w + A+.*u  : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        A, u, REPLACE);
    my_timer.stop();
    std::cout << "w<!M,replace> := w + A+.*u: " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    //===================
    // A'*x
    //===================
    std::cout << "OPTIMIZED IMPLEMENTATION: A'*u" << std::endl;
    w.clear();
    my_timer.start();
    mxv(w, NoMask(), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), u);
    my_timer.stop();
    std::cout << "w := A'+.*u                : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, NoMask(), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), u);
    my_timer.stop();
    std::cout << "w := w + A'+.*u            : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), u);
    my_timer.stop();
    std::cout << "w<M,merge> := A'+.*u       : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, M, NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), u, REPLACE);
    my_timer.stop();
    std::cout << "w<M,replace> := A+.*u      : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), u);
    my_timer.stop();
    std::cout << "w<M,merge> := w + A'+.*u   : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, M, Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), u, REPLACE);
    my_timer.stop();
    std::cout << "w<M,replace> := w + A'+.*u : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), u);
    my_timer.stop();
    std::cout << "w<!M,merge> := A'+.*u      : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, complement(M), NoAccumulate(),
        ArithmeticSemiring<double>(),
        transpose(A), u, REPLACE);
    my_timer.stop();
    std::cout << "w<!M,replace> := A'+.*u    : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), u);
    my_timer.stop();
    std::cout << "w<!M,merge> := w + A'+.*u  : " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    my_timer.start();
    mxv(w, complement(M), Plus<double>(),
        ArithmeticSemiring<double>(),
        transpose(A), u, REPLACE);
    my_timer.stop();
    std::cout << "w<!M,replace> := w + A'+.*u: " << my_timer.elapsed()
              << " msec, w.nvals = " << w.nvals() << std::endl;

    return 0;
}
