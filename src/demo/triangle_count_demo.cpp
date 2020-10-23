/*
 * GraphBLAS Template Library (GBTL), Version 3.0
 *
 * Copyright 2020 Carnegie Mellon University, Battelle Memorial Institute, and
 * Authors.
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
 * RIGHTS.
 *
 * Released under a BSD-style license, please see LICENSE file or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution.  Please see Copyright notice for non-US
 * Government use and distribution.
 *
 * DM20-0442
 */

#include <iostream>
#include <fstream>
#include <chrono>

#define GRAPHBLAS_DEBUG 1

#include <graphblas/graphblas.hpp>
#include <algorithms/triangle_count.hpp>
#include "Timer.hpp"


#include <metall/metall.hpp>



//****************************************************************************
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "ERROR: too few arguments." << std::endl;
        exit(1);
    }

    // Read the edgelist and create the tuple arrays
    std::string pathname(argv[1]);


    Timer<std::chrono::steady_clock, std::chrono::microseconds> my_timer;

    grb::IndexArrayType iL, iU, iA;
    grb::IndexArrayType jL, jU, jA;
    uint64_t num_rows = 0;
    uint64_t max_id = 0;
    uint64_t src, dst;

    my_timer.start();
    {
        std::ifstream infile(pathname);
        while (infile)
        {
            infile >> src >> dst;
            //std::cout << "Read: " << src << ", " << dst << std::endl;
            if (src > max_id) max_id = src;
            if (dst > max_id) max_id = dst;

            if (src < dst)
            {
                iA.push_back(src);
                jA.push_back(dst);

                iU.push_back(src);
                jU.push_back(dst);
            }
            else if (dst < src)
            {
                iA.push_back(src);
                jA.push_back(dst);

                iL.push_back(src);
                jL.push_back(dst);
            }
            // else ignore self loops

            ++num_rows;
        }
    }
    my_timer.stop();
    std::cout << "Elapsed read time: " << my_timer.elapsed() << " usec." << std::endl;

    std::cout << "Read " << num_rows << " rows." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    // sort the
    using DegIdx = std::tuple<grb::IndexType,grb::IndexType>;
    my_timer.start();
    std::vector<DegIdx> degrees(max_id + 1);
    for (grb::IndexType idx = 0; idx <= max_id; ++idx)
    {
        degrees[idx] = {0UL, idx};
    }

    {
        std::ifstream infile(pathname);
        while (infile)
        {
            infile >> src >> dst;
            if (src != dst)
            {
                std::get<0>(degrees[src]) += 1;
            }
        }
    }

    std::sort(degrees.begin(), degrees.end(),
              [](DegIdx a, DegIdx b) { return std::get<0>(b) < std::get<0>(a); });

    //relabel
    for (auto &idx : iA) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : jA) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : iL) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : jL) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : iU) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : jU) { idx = std::get<1>(degrees[idx]); }

    my_timer.stop();
    std::cout << "Elapsed sort/relabel time: " << my_timer.elapsed() << " usec." << std::endl;

    grb::IndexType idx(0);
    for (auto&& row : degrees)
    {
        std::cout << idx << " <-- " << std::get<1>(row)
                  << ": deg = " << std::get<0>(row) << std::endl;
        idx++;
    }

    using T = int32_t;
    using Metall_MatType = grb::Matrix<T, metall::manager::allocator_type<char>>;
    //using MatType = grb::Matrix<T, grb::DirectedMatrixTag>;

    grb::IndexType NUM_NODES(max_id + 1);
    std::vector<T> v(iA.size(), 1);

    {
        metall::manager manager(metall::create_only, "/tmp/kaushik/datastore");
        Metall_MatType *L = manager.construct<Metall_MatType>("gbtl_vov_matrix")
                        ( NUM_NODES, NUM_NODES, manager.get_allocator());
        L->build(iL.begin(), jL.begin(), v.begin(), iL.size());

        //Metall_MatType A(NUM_NODES, NUM_NODES);
        //Metall_MatType L(NUM_NODES, NUM_NODES);
        //Metall_MatType U(NUM_NODES, NUM_NODES);
        //A->build(iA.begin(), jA.begin(), v.begin(), iA.size());
        //U->build(iU.begin(), jU.begin(), v.begin(), iU.size());
    }

    {
        metall::manager manager(metall::open_only, "/tmp/kaushik/datastore");
        Metall_MatType *L = manager.find<Metall_MatType>("gbtl_vov_matrix").first;

        std::cout << "Running algorithm(s)..." << std::endl;
        T count(0);

        my_timer.start();
        count = algorithms::triangle_count_masked(*L);
        my_timer.stop();

        std::cout << "# triangles (C<L> = L +.* L'; #=|C|) = " << count << std::endl;
        std::cout << "Elapsed time: " << my_timer.elapsed() << " usec." << std::endl;
    }

/*
    // Perform triangle counting with different algorithms
    //===================
    my_timer.start();
    count = algorithms::triangle_count(A);
    my_timer.stop();

    std::cout << "# triangles (L,U=split(A); B=LL'; C=A.*B; #=|C|/2) = " << count << std::endl;
    std::cout << "Elapsed time: " << my_timer.elapsed() << " usec." << std::endl;

    //===================
    my_timer.start();
    count = algorithms::triangle_count_masked(L, U);
    my_timer.stop();

    std::cout << "# triangles (C<L> = L +.* U; #=|C|) = " << count << std::endl;
    std::cout << "Elapsed time: " << my_timer.elapsed() << " usec." << std::endl;


    //===================
    my_timer.start();
    count = algorithms::triangle_count_masked_noT(L);
    my_timer.stop();

    std::cout << "# triangles (C<L> = L +.* L; #=|C|) = " << count << std::endl;
    std::cout << "Elapsed time: " << my_timer.elapsed() << " usec." << std::endl;

    //===================
    my_timer.start();
    count = algorithms::triangle_count_newGBTL(L, U);
    my_timer.stop();

    std::cout << "# triangles (B=LU; C=L.*B; #=|C|) = " << count << std::endl;
    std::cout << "Elapsed time: " << my_timer.elapsed() << " usec." << std::endl;
*/
    return 0;
}
