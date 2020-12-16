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
#include "Timer.hpp"

#include <graphblas/graphblas.hpp>
#include <algorithms/triangle_count.hpp>
#include <algorithms/sssp.hpp>
#include <algorithms/bfs.hpp>
#include <algorithms/cluster_louvain.hpp>
#include <algorithms/k_truss.hpp>

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

    std::cout << "Read " << num_rows << " rows." << std::endl;
    std::cout << "#Nodes = " << (max_id + 1) << std::endl;

    // sort the
    using DegIdx = std::tuple<grb::IndexType,grb::IndexType>;
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
   
    for (auto &idx : iL) { idx = std::get<1>(degrees[idx]); }
    for (auto &idx : jL) { idx = std::get<1>(degrees[idx]); }

    grb::IndexType NUM_NODES(max_id + 1);
    using T = int32_t;
    std::vector<T> v(iA.size(), 1);

    /// @todo change scalar type to unsigned int or grb::IndexType
    using MatType = grb::Matrix<T>;
    MatType A(NUM_NODES, NUM_NODES);
    A.build(iA.begin(), jA.begin(), v.begin(), iA.size());

    my_timer.stop();
    std::cout << "Graph Construction time: \t" << my_timer.elapsed() << " usec." << std::endl;

    //=========================Triangle Counting================================
     T count(0);
    my_timer.start();
    count = algorithms::triangle_count(A);
    my_timer.stop();
    std::cout << "TC Algorithm time: \t\t" << my_timer.elapsed() 
                << " usec. Triangles=" << count << std::endl;

    //=========================single BFS================================
    my_timer.start();
    grb::Vector<T> parent_list(NUM_NODES);
    grb::Vector<T> root(NUM_NODES);
    root.setElement(iA.front(), jA.front());
    algorithms::bfs(A, root, parent_list);
    //grb::print_vector(std::cout, parent_list, "Parent list for root at vertex 3");
    my_timer.stop();
    std::cout << "BFS Algorithm time: \t\t" << my_timer.elapsed() << " usec." << std::endl;

/*    //=========================one SSSP================================
    my_timer.start();
    grb::Vector<double> path(NUM_NODES);
    path.setElement(iA.front(), jA.front());
    algorithms::sssp(A, path);
    // grb::print_vector(std::cout, path, "single SSSP results");
    my_timer.stop();
    std::cout << "SSSP algorithm time: \t\t" << my_timer.elapsed() << " usec." << std::endl;
*/
    //=========================2-K-truss================================
    my_timer.start();
    auto Eout2 = algorithms::k_truss(A, 2); 
    my_timer.stop();
    std::cout << "2-trusses algorithm time: \t" << my_timer.elapsed() << " usec." << std::endl;



    return 0;
}
