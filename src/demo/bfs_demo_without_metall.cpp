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
#include <algorithms/bfs.hpp>
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

    my_timer.start();

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
    


    /*    grb::IndexType idx(0);
    for (auto&& row : degrees)
    {
        std::cout << idx << " <-- " << std::get<1>(row)
                  << ": deg = " << std::get<0>(row) << std::endl;
        idx++;
    } */
    my_timer.stop();
    std::cout << "Elapsed sort/relabel time: " << my_timer.elapsed() << " usec." << std::endl;

    my_timer.start();


    grb::IndexType NUM_NODES(max_id + 1);

    /// @todo change scalar type to unsigned int or grb::IndexType
    using T = grb::IndexType;
    using GBMatrix = grb::Matrix<T, grb::DirectedMatrixTag>;
    //T const INF(std::numeric_limits<T>::max());

    std::vector<T> v(iA.size(), 1);


    GBMatrix G_tn(NUM_NODES, NUM_NODES);
    G_tn.build(iA.begin(), jA.begin(), v.begin(), iA.size());
   // grb::print_matrix(std::cout, G_tn, "Graph adjacency matrix:");


    my_timer.stop();
    std::cout << "Graph Construction time: " << my_timer.elapsed() << " usec." << std::endl;
    std::cout << iA.front() << jA.front() << std::endl;



    my_timer.start();

    // Perform a single BFS
    grb::Vector<T> parent_list(NUM_NODES);
    grb::Vector<T> root(NUM_NODES);
    root.setElement(iA.front(), jA.front());
    algorithms::bfs(G_tn, root, parent_list);
    //grb::print_vector(std::cout, parent_list, "Parent list for root at vertex 3");

    my_timer.stop();
    std::cout << "Algorithm time: " << my_timer.elapsed() << " usec." << std::endl;
    return 0;
}
